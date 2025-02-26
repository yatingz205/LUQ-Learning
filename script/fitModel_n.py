# Generate data, define likelihood and diagnostic functions
import os
import sys
import time
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp


# General setup
sn = int(sys.argv[1])
seed = int(sys.argv[2])
n = int(sys.argv[3])
n_sim = 1000
run = 'sn=' + str(sn) + '_seed=' + str(seed) + '_n=' + str(n)
print('run:', run)

check_file = "./estData/alpha_opt_" + run + ".npy"
if os.path.exists(check_file):
    print(f"Job skipped: {check_file} already exists.")
    sys.exit(0) 

exec(open("likFuncs_n.py").read())

# Optimization setup
print('Printing diagnostics for truth')
print_diagnostics(beta, alpha, lambda_)

seeds = [42, 43, 44, 45, 46]
nll_opt = np.inf
start_time = time.time()

# Optimize over multiple seeds
for seed in seeds:
    print('Starting seed:', seed)
    tf.random.set_seed(seed)  # Updated method

    # Generate initial values
    with tf.device('/GPU:0'):  # Run all tensor operations on GPU
        beta_est_init = tf.random.normal(beta.shape, mean=0., stddev=0.1, dtype=tf.float32)
        beta_est = tf.Variable(beta_est_init)
        alpha0_est = tf.random.normal(alpha0.shape, mean=0., stddev=0.1, dtype=tf.float32)
        alpha0_est = tf.sort(alpha0_est)
        alpha_temp_est = tf.random.gamma(alpha_temp.shape, alpha=1, beta=10, dtype=tf.float32)
        alpha_est_init = tf.concat([alpha0_est, tf.expand_dims(alpha_temp_est, axis=1)], axis=1)
        alpha_est = tf.Variable(alpha_est_init)
        lambda_est_init = tf.random.gamma(lambda_.shape, alpha=1, beta=10, dtype=tf.float32)
        lambda_est = tf.Variable(lambda_est_init)

    print('Printing diagnostics for initial values')
    print_diagnostics(beta_est, alpha_est, lambda_est)

    # Optimization setup
    loss_fn = lambda: NLL_func(beta_est, alpha_est, lambda_est)
    var_list = [beta_est, alpha_est, lambda_est]
    lbfgs_iter = tf.Variable(0, dtype=tf.int32)
    shapes = tf.shape_n(var_list)
    n_tensors = len(shapes)
    count, idx, part = 0, [], []
    for i, shape in enumerate(shapes):
        n2 = np.prod(shape)
        idx.append(tf.reshape(tf.range(count, count + n2, dtype=tf.int32), shape))
        part.extend([i] * n2)
        count += n2
    part = tf.constant(part, dtype=tf.int32)

    # Initial SGD optimization
    opt = tf.keras.optimizers.legacy.SGD(learning_rate=1e-5)
    for _ in range(500):
        opt.minimize(loss_fn, var_list)
    print('Printing diagnostics after initial optimization')
    print_diagnostics(beta_est, alpha_est, lambda_est)


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
    results = tfp.optimizer.lbfgs_minimize(f, init_params, max_iterations=1000)
    
    print('Printing diagnostics after LBFGS optimization')
    print_diagnostics(beta_est, alpha_est, lambda_est)

    # Update parameters
    new_nll = loss_fn()
    if new_nll < nll_opt:
        print('Updating parameters')
        beta_opt, alpha_opt, lambda_opt = beta_est, alpha_est, lambda_est
        nll_opt = new_nll


#saving optimized parms
end_time = time.time()
elapsed_time = end_time-start_time
print('printing diagnostics for fitted model')
print_diagnostics(beta_opt, alpha_opt, lambda_opt)

grads = grad_func(beta_est, alpha_est, lambda_est)
lInf_grads = max(tf.reduce_max(grads[0]), tf.reduce_max(grads[1]), tf.reduce_max(grads[2]))
total_params = tf.cast(tf.size(beta) + tf.size(alpha) + tf.size(lambda_), tf.float32)
l1_error = (tf.reduce_sum(tf.math.abs(beta_est-beta)) +
            tf.reduce_sum(tf.math.abs(alpha_est-alpha)) +
            tf.reduce_sum(tf.math.abs(lambda_est-lambda_)))/total_params
lInf_error = max(tf.reduce_max(tf.math.abs(beta_est-beta)),
                    tf.reduce_max(tf.math.abs(alpha_est-alpha)),
                    tf.reduce_max(tf.math.abs(lambda_est-lambda_)))
errors = tf.stack((lInf_grads, l1_error, lInf_error))

np.save('estData/errors_' + run + '.npy', errors.numpy())
np.save('estData/beta_opt_' + run + '.npy', beta_opt.numpy())
np.save('estData/alpha_opt_' + run + '.npy', alpha_opt.numpy())
np.save('estData/lambda_opt_' + run + '.npy', lambda_opt.numpy()) 
np.save('estData/time_' + run + '.npy', elapsed_time) 