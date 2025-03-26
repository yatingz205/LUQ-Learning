# Generate data, define likelihood and diagnostic functions
import os
import sys
import time
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

# General setup
seed, K, n = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
n_sim = 1000
run = 'seed=' + str(seed) + '_K=' + str(K) + '_n=' + str(n)
print('run:', run)

check_file = "./estData/alpha_opt_" + run + ".npy"
if os.path.exists(check_file):
    print(f"Job skipped: {check_file} already exists.")
    sys.exit(0)

exec(open("likFuncs_K.py").read())

# Optimization setup
print('Printing diagnostics for truth')
print_diagnostics(beta, alpha)

seeds = list(range(42, 52))
nll_opt = np.inf
report = 50
start_time = time.time()

# Optimize over multiple seeds
for seed in seeds:
    print('Starting seed:', seed)
    tf.random.set_seed(seed)  # Updated method

    # Generate initial values
    with tf.device('/GPU:0'):  # Ensure tensors are created on the GPU
        beta_est = tf.Variable(tf.random.normal(beta.shape, mean=0., stddev=0.1, dtype=tf.float32))
        alpha0_est = tf.random.normal(alpha0.shape, mean=0., stddev=0.1, dtype=tf.float32)
        alpha0_est = tf.sort(alpha0_est, axis=0)
        alpha1_est = tf.random.gamma(alpha1.shape, alpha=1, beta=10, dtype=tf.float32)
        alpha_est = tf.Variable(tf.concat((alpha0_est, tf.expand_dims(alpha1_est, axis=0)), axis=0))

    # Initial SGD optimization
    loss_fn = lambda: NLL_func(beta_est, alpha_est)
    var_list = [beta_est, alpha_est]
    opt = tf.keras.optimizers.legacy.SGD()
    for _ in range(250):
        opt.minimize(loss_fn, var_list)

    # Optimization setup for LBFGS
    lbfgs_iter = tf.Variable(0, dtype=tf.int32)
    shapes = tf.shape_n(var_list)
    n_tensors = len(shapes)
    count, idx, part = 0, [], []

    # Generate index tensors for dynamic stitching
    with tf.device('/CPU:0'):  # Use CPU for unsupported slicing/indexing operations
        for i, shape in enumerate(shapes):
            n2 = np.prod(shape)
            idx.append(tf.reshape(tf.range(count, count + n2, dtype=tf.int32), shape))
            part.extend([i] * n2)
            count += n2
        part = tf.constant(part, dtype=tf.int32)

    # Define objective function callable for LBFGS
    def f(params_1d):
        params = tf.dynamic_partition(params_1d, part, n_tensors)
        for i, (shape, param) in enumerate(zip(shapes, params)):
            var_list[i].assign(tf.reshape(param, shape))

        with tf.device('/GPU:0'):  
            with tf.GradientTape() as tape:
                nll = loss_fn()
            grads = tape.gradient(nll, var_list)
            grads = tf.dynamic_stitch(idx, grads)
            lbfgs_iter.assign_add(1)
        return nll, grads

    # LBFGS optimization
    lbfgs_iter.assign(0)
    init_params = tf.dynamic_stitch(idx, var_list)
    results = tfp.optimizer.lbfgs_minimize(f, init_params, max_iterations=1000, tolerance=1e-4)
    print_diagnostics(beta_est, alpha_est)

    # Update parameters
    new_nll = loss_fn()
    if new_nll < nll_opt:
        beta_opt, alpha_opt = beta_est, alpha_est
        nll_opt = new_nll


end_time = time.time()
elapsed_time = end_time-start_time
print('Printing diagnostics for fitted model')
print_diagnostics(beta_opt, alpha_opt)

grads = grad_func(beta_est, alpha_est)
theta_dim = tf.cast(tf.reduce_prod(beta_est.shape) + tf.reduce_prod(alpha_est.shape), tf.float32)
lInf_grads = max(tf.reduce_max(grads[0]), tf.reduce_max(grads[1]))
mae_error = (tf.reduce_sum(tf.abs(beta_est - beta)) + tf.reduce_sum(tf.abs(alpha_est - alpha))) / theta_dim
lInf_error = max(tf.reduce_max(tf.abs(beta_est - beta)), tf.reduce_max(tf.abs(alpha_est - alpha)))
errors = tf.stack((lInf_grads, mae_error, lInf_error))

np.save('estData/errors_' + run + '.npy', errors.numpy())
np.save('estData/beta_opt_' + run + '.npy', beta_opt.numpy())
np.save('estData/alpha_opt_' + run + '.npy', alpha_opt.numpy())
np.save('estData/time_' + run + '.npy', elapsed_time) 