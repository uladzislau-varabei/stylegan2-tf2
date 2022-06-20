import tensorflow as tf

from tf_utils import fp32, maybe_scale_loss, maybe_custom_unscale_grads


# Note: all losses are decorated with tf.function during their initialization.

def select_G_loss_fn(loss_name, use_xla=False):
    losses = {
        'G_wgan'.lower(): G_wgan,
        'G_logistic'.lower(): G_logistic_saturating,
        'G_logistic_ns'.lower(): G_logistic_ns,
        'G_logistic_ns_pathreg'.lower(): G_logistic_ns_pathreg
    }
    assert loss_name.lower() in losses.keys(), \
        f"Generator loss function {loss_name} is not supported, see 'select_G_loss_fn'"
    return tf.function(losses[loss_name.lower()], jit_compile=use_xla)


def select_D_loss_fn(loss_name, use_xla=False):
    losses = {
        'D_wgan'.lower(): D_wgan,
        'D_wgan_gp'.lower(): D_wgan_gp,
        'D_logistic'.lower(): D_logistic,
        'D_logistic_simplegp'.lower(): D_logistic_simplegp
    }
    assert loss_name.lower() in losses.keys(), \
        f"Discriminator loss function {loss_name} is not supported, see 'select_D_loss_fn'"
    return tf.function(losses[loss_name.lower()], jit_compile=use_xla)


def tf_sum(x, axis=None):
    return tf.reduce_sum(x, axis=axis)


def tf_mean(x, axis=None):
    return tf.reduce_mean(x, axis=axis)


def create_zero_tensor(dtype=None):
    return tf.zeros([1], dtype=tf.float32 if dtype is None else dtype)


#----------------------------------------------------------------------------
# WGAN & WGAN-GP loss functions.

def G_wgan(G, D, optimizer, batch_size, evaluate_loss, evaluate_reg, write_summary, step, **kwargs):
    if evaluate_loss:
        latents = G.generate_latents(batch_size)
        fake_images = G(latents, training=True)
        fake_scores = fp32(D(fake_images, training=True))
        loss = -fake_scores
    else:
        latents = create_zero_tensor(G.model_compute_dtype)
        fake_images = create_zero_tensor(G.model_compute_dtype)
        fake_scores = create_zero_tensor()
        loss = create_zero_tensor()

    with tf.name_scope('Loss/G_WGAN'):
        if write_summary:
            tf.summary.scalar('Total', loss, step=step)

    return tf_mean(loss), 0.0


def D_wgan(G, D, optimizer, batch_size, evaluate_loss, evaluate_reg, real_images, write_summary, step,
    wgan_epsilon = 0.001,  # Weight for the epsilon term, \epsilon_{drift}
    **kwargs):

    if evaluate_loss:
        latents = G.generate_latents(batch_size)
        fake_images = G(latents, training=True)
        fake_scores = fp32(D(fake_images, training=True))
        real_scores = fp32(D(real_images, training=True))
        fakes_loss = tf_mean(fake_scores)
        reals_loss = tf_mean(real_scores)
        loss = fakes_loss - reals_loss
    else:
        latents = create_zero_tensor(G.model_compute_dtype)
        fake_images = create_zero_tensor(G.model_compute_dtype)
        fake_scores = create_zero_tensor()
        real_scores = create_zero_tensor()
        fakes_loss = create_zero_tensor()
        reals_loss = create_zero_tensor()
        loss = create_zero_tensor()

    if evaluate_reg:
        # Epsilon penalty
        epsilon_penalty = tf_mean(tf.square(real_scores))
        reg = wgan_epsilon * epsilon_penalty
    else:
        wgan_epsilon = create_zero_tensor()
        reg = create_zero_tensor()

    with tf.name_scope('Loss/D_WGAN'):
        if write_summary:
            tf.summary.scalar('FakeScores', tf_mean(fake_scores), step=step)
            tf.summary.scalar('RealScores', tf_mean(real_scores), step=step)
            tf.summary.scalar('FakePart', fakes_loss, step=step)
            tf.summary.scalar('RealPart', reals_loss, step=step)
            tf.summary.scalar('Total', loss, step=step)
            if evaluate_reg:
                tf.summary.scalar('EpsilonPenalty', epsilon_penalty, step=step)
                tf.summary.scalar('Reg', tf_mean(reg), step=step)

    return tf_mean(loss), tf_mean(reg)


def D_wgan_gp(G, D, optimizer, batch_size, evaluate_loss, evaluate_reg, real_images, write_summary, step,
    wgan_lambda  = 10.0,   # Weight for the gradient penalty term
    wgan_epsilon = 0.001,  # Weight for the epsilon term, \epsilon_{drift}
    wgan_target  = 1.0,    # Target value for gradient magnitudes
    **kwargs):

    if evaluate_loss:
        latents = G.generate_latents(batch_size)
        fake_images = G(latents, training=True)
        fake_scores = fp32(D(fake_images, training=True))
        real_scores = fp32(D(real_images, training=True))
        loss = fake_scores - real_scores
    else:
        latents = create_zero_tensor(G.model_compute_dtype)
        fake_images = create_zero_tensor(G.model_compute_dtype)
        fake_scores = create_zero_tensor()
        real_scores = create_zero_tensor()
        loss = create_zero_tensor()

    if evaluate_reg:
        # Gradient penalty
        alpha = tf.random.uniform(shape=[batch_size, 1, 1, 1], minval=0.0, maxval=1.0, dtype=real_images.dtype)
        mixed_images = alpha * real_images + (1.0 - alpha) * fake_images
        with tf.GradientTape(watch_accessed_variables=False) as tape_gp:
            tape_gp.watch(mixed_images)
            mixed_scores = fp32(D(mixed_images, training=True))
            mixed_loss = maybe_scale_loss(tf_sum(mixed_scores), optimizer)
        gp_grads = fp32(tape_gp.gradient(mixed_loss, mixed_images))
        # Default grads unscaling doesn't work inside this function, though it is ok to use it inside train steps
        gp_grads = maybe_custom_unscale_grads(gp_grads, mixed_images, optimizer)
        gp_grads_norm = tf.sqrt(tf_sum(tf.square(gp_grads), axis=[1, 2, 3]))
        grads_penalty = (gp_grads_norm - wgan_target) ** 2
        reg = (wgan_lambda / (wgan_target ** 2)) * grads_penalty

        # Epsilon penalty
        epsilon_penalty = tf.square(real_scores)
        reg += wgan_epsilon * epsilon_penalty
    else:
        reg = create_zero_tensor()

    with tf.name_scope('Loss/D_WGAN-GP'):
        if write_summary:
            tf.summary.scalar('FakeScores', tf_mean(fake_scores), step=step)
            tf.summary.scalar('RealScores', tf_mean(real_scores), step=step)
            tf.summary.scalar('Total', tf_mean(loss), step=step)
            if evaluate_reg:
                tf.summary.scalar('GradsPenalty', tf_mean(grads_penalty), step=step)
                tf.summary.scalar('EpsilonPenalty', tf_mean(epsilon_penalty), step=step)
                tf.summary.scalar('Reg', tf_mean(reg), step=step)

    return tf_mean(loss), tf_mean(reg)


#----------------------------------------------------------------------------
# New loss functions used by StyleGAN.
# Loss functions advocated by the paper "Which Training Methods for GANs do actually Converge?"

def G_logistic_saturating(G, D, optimizer, batch_size, write_summary, step, **kwargs):
    latents = G.generate_latents(batch_size)
    fake_images = G(latents, training=True)
    fake_scores = fp32(D(fake_images, training=True))
    loss = -tf.nn.softplus(fake_scores) # log(1 - logistic(fake_scores))

    with tf.name_scope('Loss/G_logistic_saturating'):
        if write_summary:
            tf.summary.scalar('FakeScores', tf_mean(fake_scores), step=step)
            tf.summary.scalar('Total', tf_mean(loss), step=step)

    return tf_mean(loss), 0.0


def G_logistic_ns(G, D, optimizer, batch_size, evaluate_loss, evaluate_reg, write_summary, step, **kwargs):
    if evaluate_loss:
        latents = G.generate_latents(batch_size)
        fake_images = G(latents, training=True)
        fake_scores = fp32(D(fake_images, training=True))
        loss = tf.nn.softplus(-fake_scores) # -log(logistic(fake_scores))
    else:
        latents = create_zero_tensor(G.model_compute_dtype)
        fake_images = create_zero_tensor(G.model_compute_dtype)
        fake_scores = create_zero_tensor()
        loss = create_zero_tensor()

    with tf.name_scope('Loss/G_logistic_nonsaturating'):
        if write_summary:
            tf.summary.scalar('FakeScores', tf_mean(fake_scores), step=step)
            tf.summary.scalar('Total', tf_mean(loss), step=step)

    return tf_mean(loss), 0.0


def D_logistic(G, D, optimizer, batch_size, real_images, evaluate_loss, evaluate_reg, write_summary, step, **kwargs):
    if evaluate_loss:
        latents = G.generate_latents(batch_size)
        fake_images = G(latents, training=True)
        fake_scores = fp32(D(fake_images, training=True))
        real_scores = fp32(D(real_images, training=True))
        fakes_loss = tf.nn.softplus(fake_scores) # -log(1 - logistic(fake_scores))
        reals_loss = tf.nn.softplus(-real_scores) # -log(logistic(real_scores))
        loss = fakes_loss + reals_loss
    else:
        latents = create_zero_tensor(G.model_compute_dtype)
        fake_images = create_zero_tensor(G.model_compute_dtype)
        fake_scores = create_zero_tensor()
        real_scores = create_zero_tensor()
        fakes_loss = create_zero_tensor()
        reals_loss = create_zero_tensor()
        loss = create_zero_tensor()

    with tf.name_scope('Loss/D_logistic'):
        if write_summary:
            tf.summary.scalar('FakeScores', tf_mean(fake_scores), step=step)
            tf.summary.scalar('RealScores', tf_mean(real_scores), step=step)
            tf.summary.scalar('FakePart', tf_mean(fakes_loss), step=step)
            tf.summary.scalar('RealPart', tf_mean(reals_loss), step=step)
            tf.summary.scalar('Total', tf_mean(loss), step=step)

    return tf_mean(loss), 0.0


def D_logistic_simplegp(G, D, optimizer, batch_size, real_images, evaluate_loss, evaluate_reg, write_summary, step,
                        r1_gamma=10.0, r2_gamma=0.0, **kwargs):
    if evaluate_loss:
        latents = G.generate_latents(batch_size)
        fake_images = G(latents, training=True)
        fake_scores = fp32(D(fake_images, training=True))
        real_scores = fp32(D(real_images, training=True))
        fakes_loss = tf.nn.softplus(fake_scores) # -log(1 - logistic(fake_scores))
        reals_loss = tf.nn.softplus(-real_scores) # -log(logistic(real_scores))
        loss = fakes_loss + reals_loss
    else:
        latents = create_zero_tensor(G.model_compute_dtype)
        fake_images = create_zero_tensor(G.model_compute_dtype)
        fake_scores = create_zero_tensor()
        real_scores = create_zero_tensor()
        fakes_loss = create_zero_tensor()
        reals_loss = create_zero_tensor()
        loss = create_zero_tensor()

    if evaluate_reg:
        reg = 0.0
        use_r1_penalty = r1_gamma > 0.0
        use_r2_penalty = r2_gamma > 0.0
        with tf.GradientTape(watch_accessed_variables=False, persistent=True) as tape_gp:
            # r1 penalty
            if use_r1_penalty:
                tape_gp.watch(real_images)
                real_scores = fp32(D(real_images, training=True))
            else:
                real_scores = create_zero_tensor()
            # r2 penalty
            if use_r2_penalty:
                latents = G.generate_latents(batch_size)
                fake_images = G(latents, training=True)
                tape_gp.watch(fake_images)
                fake_scores = fp32(D(fake_images, training=True))
            else:
                fake_images = create_zero_tensor(G.model_compute_dtype)
                fake_scores = create_zero_tensor()
            # optional loss scaling
            real_loss = maybe_scale_loss(tf_sum(real_scores), optimizer) if use_r1_penalty else 0.0
            fake_loss = maybe_scale_loss(tf_sum(fake_scores), optimizer) if use_r2_penalty else 0.0

        if use_r1_penalty:
            real_grads = fp32(tape_gp.gradient(real_loss, real_images))
            real_grads = maybe_custom_unscale_grads(real_grads, real_images, optimizer)
            r1_penalty = tf_sum(tf.square(real_grads), axis=[1, 2, 3])
            reg += r1_penalty * (r1_gamma * 0.5)
        else:
            r1_penalty = 0.0

        if use_r2_penalty:
            fake_grads = fp32(tape_gp.gradient(fake_loss, fake_images))
            fake_grads = maybe_custom_unscale_grads(fake_grads, fake_images, optimizer)
            r2_penalty = tf_sum(tf.square(fake_grads), axis=[1, 2, 3])
            reg += r2_penalty * (r2_gamma * 0.5)
        else:
            r2_penalty = 0.0
    else:
        use_r1_penalty = r1_gamma > 0.0
        use_r2_penalty = r2_gamma > 0.0
        r1_penalty = create_zero_tensor()
        r2_penalty = create_zero_tensor()
        reg = create_zero_tensor()

    with tf.name_scope('Loss/D_logistic_simpleGP'):
        if write_summary:
            if evaluate_loss:
                tf.summary.scalar('FakeScores', tf_mean(fake_scores), step=step)
                tf.summary.scalar('RealScores', tf_mean(real_scores), step=step)
                tf.summary.scalar('FakePart', tf_mean(fakes_loss), step=step)
                tf.summary.scalar('RealPart', tf_mean(reals_loss), step=step)
                tf.summary.scalar('Total', tf_mean(loss), step=step)
            if evaluate_reg:
                if use_r1_penalty:
                    tf.summary.scalar('R1Penalty', tf_mean(r1_penalty), step=step)
                if use_r2_penalty:
                    tf.summary.scalar('R2Penalty', tf_mean(r2_penalty), step=step)
                tf.summary.scalar('Reg', tf_mean(reg), step=step)

    return tf_mean(loss), tf_mean(reg)


#----------------------------------------------------------------------------
# Non-saturating logistic loss with path length regularizer from the paper
# "Analyzing and Improving the Image Quality of StyleGAN", Karras et al. 2019

# Only create losses vars after GPU is initialized with some memory settings
def create_loss_vars():
    class LossVars:
        # Currently only G_logistic_ns_pathreg loss requires extra variable
        pl_mean_var = tf.Variable(initial_value=0.0, trainable=False, name='pl_mean', dtype=tf.float32, shape=[])
    return LossVars()


def G_logistic_ns_pathreg(G, D, optimizer, batch_size, evaluate_loss, evaluate_reg, loss_vars, write_summary, step,
                          pl_batch_shrink=2, pl_decay=0.01, pl_weight=2.0, **kwargs):
    if evaluate_loss:
        latents = G.generate_latents(batch_size)
        fake_images, fake_dlatents = G(latents, training=True, return_dlatents=True)
        fake_scores = fp32(D(fake_images, training=True))
        loss = tf.nn.softplus(-fake_scores) # -log(logistic(fake_scores))
    else:
        latents = create_zero_tensor(G.model_compute_dtype)
        fake_images, fake_dlatents = create_zero_tensor(G.model_compute_dtype), create_zero_tensor(G.model_compute_dtype)
        fake_scores = create_zero_tensor()
        loss = create_zero_tensor()

    # Path length regularization
    if evaluate_reg:
        with tf.GradientTape(watch_accessed_variables=False) as pl_tape:
            pl_latents = G.generate_latents(batch_size // pl_batch_shrink)
            pl_tape.watch(pl_latents)
            # 1. Evaluate the regularization term using smaller minibatch to conserve memory.
            pl_images, pl_dlatents = G(pl_latents, training=True, return_dlatents=True)
            # 2. Compute |J*y|
            pl_images_shape = tf.shape(pl_images)
            num_pixels = tf.reduce_prod(pl_images_shape[1:]) / tf.reduce_min(pl_images_shape[1:]) # Supports NCHW and NHWC formats
            pl_noise = tf.random.normal(pl_images_shape, mean=0.0, stddev=1.0, dtype=tf.float32) / fp32(num_pixels)
            Jy = maybe_scale_loss(tf_sum(fp32(pl_images) * pl_noise), optimizer)
        pl_grads = fp32(pl_tape.gradient(Jy, pl_dlatents))
        pl_grads = maybe_custom_unscale_grads(pl_grads, pl_dlatents, optimizer)
        pl_lengths = tf.sqrt(tf_mean(tf_sum(tf.square(pl_grads), axis=2), axis=1))

        nans = tf.math.count_nonzero(~tf.math.is_finite(pl_lengths))
        if nans == 0:
            # 3. Track exponential moving average of |J*y|.
            tf.print('Current pl mean:', loss_vars.pl_mean_var)
            pl_mean = loss_vars.pl_mean_var + pl_decay * (tf_mean(pl_lengths) - loss_vars.pl_mean_var)
            loss_vars.pl_mean_var.assign(pl_mean)

            # 4. Calculate (|J*y| - a) ^ 2.
            pl_penalty = tf.square(pl_lengths - pl_mean)

            # 5. Apply weight.
            # Note: the division in pl_noise decreases the weight by num_pixels, and the reduce_mean
            # in pl_lengths decreases it by num_affine_layers. The effective weight then becomes:
            #    gamma_pl = pl_weight / num_pixels / num_affine_layers =
            #             = 2 / (r^2) / (log2(r) * 2 - 2) =
            #             = ln(2) / (r^2 * (ln(r) - ln(2)))
        else:
            pl_mean = 0.0
            pl_penalty = 0.0
        reg = pl_weight * pl_penalty
        tf.print('pl_mean:', pl_mean)
    else:
        pl_mean = 0.0
        pl_penalty = 0.0
        reg = create_zero_tensor()

    with tf.name_scope('Loss/G_logistic_ns_pathreg'):
        if write_summary:
            if evaluate_loss:
                tf.summary.scalar('FakeScores', tf_mean(fake_scores), step=step)
                tf.summary.scalar('Total', tf_mean(loss), step=step)
            if evaluate_reg:
                tf.summary.scalar('PLMean', pl_mean, step=step)
                tf.summary.scalar('PLPenalty', tf_mean(pl_penalty), step=step)
                tf.summary.scalar('PLReg', tf_mean(reg), step=step)

    return tf_mean(loss), tf_mean(reg)
