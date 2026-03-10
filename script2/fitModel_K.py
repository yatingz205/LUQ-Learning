# Generate data, define likelihood and diagnostic functions
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import sys
import time
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

seed = int(sys.argv[1])
n_sim = int(sys.argv[2])
n = int(sys.argv[3])
K = sys.argv[4]
run = 'seed=' + str(seed) + '_K=' + str(K) + '_n=' + str(n)
print('run:', run)

batch_size = 1200

#check_file = "./estData/alpha_opt_" + run + ".npy"
#if os.path.exists(check_file):
#    print(f"Job skipped: {check_file} already exists.")
#    sys.exit(0)

exec(open("likFuncs_K.py").read())
print('Printing diagnostics for truth')
print_diagnostics(beta, alpha)


seeds = [42, 43, 44, 45, 46]
nll_opt = np.inf
beta_opt, alpha_opt = None, None
start_time = time.time()

# Optimize over multiple seeds
for seed in seeds:
    print('Starting seed:', seed)
    tf.random.set_seed(seed)  
    beta_est = tf.Variable(tf.random.normal(beta.shape, mean=0., stddev=1., dtype=tf.float32))
    base = 0.75 * tf.range(1, alpha0.shape[0] + 1, dtype=tf.float32)  # shape (2,)
    base = tf.tile(base[:, tf.newaxis], [1, alpha0.shape[1]])  # shape (2, K)
    alpha0_est = base + tf.random.normal(alpha0.shape, mean=0., stddev=0.1)
    alpha0_est = tf.sort(alpha0_est, axis=0)
    alpha1_est = tf.abs(tf.random.normal(alpha1.shape, mean=0.4, stddev=0.2, dtype=tf.float32))
    alpha_est = tf.Variable(tf.concat((alpha0_est, tf.expand_dims(alpha1_est, axis=0)), axis=0))

    # Initial SGD optimization
    def loss_fn():
        val = NLL_func(beta_est, alpha_est)
        return val if tf.math.is_finite(val) else tf.constant(1e9, dtype=tf.float32)
    
    var_list = [beta_est, alpha_est]
    opt = tf.keras.optimizers.legacy.SGD(learning_rate=1e-4)
    for _ in range(200): opt.minimize(loss_fn, var_list)
    print('printing diagnostics after initial optimization')
    print_diagnostics(beta_est, alpha_est)
    
    # -------- Guard check before LBFGS --------
    with tf.GradientTape() as tape:
        nll_test = loss_fn()
    grads_test = tape.gradient(nll_test, var_list)

    bad_seed = False
    for g in grads_test:
        if g is None or not tf.reduce_all(tf.math.is_finite(g)):
            print(f"[Warning] Skipping seed {seed}: invalid gradient detected")
            bad_seed = True
            break

    if bad_seed:
        continue
    # ------------------------------------------

    # Optimization setup for LBFGS
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

    # Define objective function callable for LBFGS
    def f(params_1d):
        params = tf.dynamic_partition(params_1d, part, n_tensors)
        for i, (shape, param) in enumerate(zip(shapes, params)):
            var_list[i].assign(tf.reshape(param, shape)) 
        with tf.GradientTape() as tape:
            nll = loss_fn()
        grads = tape.gradient(nll, var_list)
        grads = tf.dynamic_stitch(idx, grads)
        lbfgs_iter.assign_add(1)
        return nll, grads

    # LBFGS optimization
    lbfgs_iter.assign(0)
    init_params = tf.dynamic_stitch(idx, var_list)
    results = tfp.optimizer.lbfgs_minimize(
        f, init_params, 
        tolerance=1e-4, 
        max_iterations=1000)
    print('printing diagnostics after LBFGS optimization')
    print_diagnostics(beta_est, alpha_est)

    # Update parameters
    new_nll = loss_fn().numpy()
    if new_nll < nll_opt:
        beta_opt, alpha_opt = beta_est, alpha_est
        nll_opt = new_nll

print("Beta error:", tf.reduce_max(tf.abs(beta_opt - beta)).numpy())
print("Alpha0 error:", tf.reduce_max(tf.abs(alpha_opt[:-1] - alpha[:-1])).numpy())
print("Alpha1 error:", tf.reduce_max(tf.abs(alpha_opt[-1] - alpha[-1])).numpy())


end_time = time.time()
elapsed_time = end_time-start_time

grads = grad_func(beta_opt, alpha_opt)
theta_dim = tf.cast(tf.reduce_prod(beta_opt.shape) + tf.reduce_prod(alpha_opt.shape), tf.float32)
lInf_grads = max(tf.reduce_max(grads[0]), tf.reduce_max(grads[1]))
mae_error = (tf.reduce_sum(tf.abs(beta_opt - beta)) + tf.reduce_sum(tf.abs(alpha_opt - alpha))) / theta_dim
lInf_error = max(tf.reduce_max(tf.abs(beta_opt - beta)), tf.reduce_max(tf.abs(alpha_opt - alpha)))
errors = tf.stack((lInf_grads, mae_error, lInf_error))

os.makedirs("estData", exist_ok=True)
np.save('estData/errors_' + run + '.npy', errors.numpy())
np.save('estData/beta_opt_' + run + '.npy', beta_opt.numpy())
np.save('estData/alpha_opt_' + run + '.npy', alpha_opt.numpy())
np.save('estData/time_' + run + '.npy', elapsed_time) 

print('Finished with elapsed time (s):', elapsed_time)