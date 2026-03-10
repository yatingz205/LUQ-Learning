#generate data, define likelihood and diagnostic functions
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
lab = sys.argv[4]
run = lab + '_seed=' + str(seed) + '_n=' + str(n)
print('run:', run)

check_file = "./estData/alpha_opt_" + run + ".npy"
if os.path.exists(check_file):
   print(f"Job skipped: {check_file} already exists.")
   sys.exit(0)

exec(open("likFuncs_n.py").read())
print('printing diagnostics for truth')
print_diagnostics(beta, alpha, lambda_)
print_diagnostics_lambda_only(lambda_)


seeds = [42, 43, 44, 45, 46]
nll_opt = np.inf
beta_opt, alpha_opt, lambda_opt = None, None, None
start_time = time.time()

#optimize over multiple seeds
for seed in seeds:
    #generate initial values
    print('starting seed:', seed)
    tf.random.set_seed(seed)
    beta_est = tf.Variable(tf.random.normal(beta.shape, mean=0., stddev=0.1, dtype=tf.float32))
    alpha0_est = tf.random.normal(alpha0.shape, mean=0., stddev=0.1, dtype=tf.float32)
    alpha0_est = tf.sort(alpha0_est)
    alpha_temp_est = tf.random.gamma(alpha_temp.shape, alpha=1, beta=10, dtype=tf.float32)
    alpha_est = tf.Variable(tf.concat((alpha0_est, alpha_temp_est[:,tf.newaxis]), axis=1))
    lambda_est = tf.Variable(tf.random.gamma(lambda_.shape, alpha=10, beta=10, dtype=tf.float32))
    print('printing diagnostics for initial values')
    print_diagnostics(beta_est, alpha_est, lambda_est)
    print_diagnostics_lambda_only(lambda_est)
    
    #optimization setup
    loss_fn = lambda: NLL_func(beta_est, alpha_est, lambda_est)
    var_list = [beta_est, alpha_est, lambda_est]
    lbfgs_iter = tf.Variable(0)
    shapes = tf.shape_n(var_list)
    n_tensors = len(shapes)
    count, idx, part = 0, [], []
    for i, shape in enumerate(shapes):
        n2 = np.product(shape)
        idx.append(tf.reshape(tf.range(count, count+n2, dtype=tf.int32), shape))
        part.extend([i]*n2)
        count += n2
    part = tf.constant(part)
    
    #initial SGD optimization
    opt = tf.keras.optimizers.SGD(learning_rate=1e-4)
    for i in range(200): weight_update = opt.minimize(loss_fn, var_list) 
    print('printing diagnostics after initial optimization')
    print_diagnostics(beta_est, alpha_est, lambda_est)
    print_diagnostics_lambda_only(lambda_est)
    
    #define objective function callable for LBFGS
    def f(params_1d):
        params = tf.dynamic_partition(params_1d, part, n_tensors)
        for i, (shape, param) in enumerate(zip(shapes, params)):
            var_list[i].assign(tf.reshape(param, shape))
        with tf.GradientTape() as tape: nll = loss_fn()
        grads = tape.gradient(nll, var_list)
        grads = tf.dynamic_stitch(idx, grads)
        lbfgs_iter.assign_add(1)
        return nll, grads
    
    #LFBGS optimization
    lbfgs_iter.assign(0) 
    init_params = tf.dynamic_stitch(idx, var_list)
    results = tfp.optimizer.lbfgs_minimize(
        f, init_params, 
        tolerance=1e-4,
        max_iterations=1000)
    print('printing diagnostics after LBFGS optimization')
    print_diagnostics(beta_est, alpha_est, lambda_est)
    print_diagnostics_lambda_only(lambda_est)
    
    print(f"L-BFGS converged: {results.converged.numpy()}")
    print(f"L-BFGS iterations: {results.num_iterations.numpy()}")
    alpha01_est, alpha1_est = alpha_est[0, :-1], alpha_est[0, -1]
    alpha02_est, alpha2_est = alpha_est[1, :-1], alpha_est[1, -1]
    lambda1_est, lambda2_est = lambda_est[0], lambda_est[1]
    print(f"Constraints: alpha1={alpha1_est:.4f}, alpha2={alpha2_est:.4f}, "
        f"lambda1={lambda1_est:.4f}, lambda2={lambda2_est:.4f}")
    print(f"alpha01 monotonicity: min_diff={tf.reduce_min(alpha01_est[1:]-alpha01_est[:-1]):.4f}")
    print(f"alpha02 monotonicity: min_diff={tf.reduce_min(alpha02_est[1:]-alpha02_est[:-1]):.4f}")
        
    #update parms
    new_nll = loss_fn()
    if new_nll < nll_opt:
        print('updating parms')
        beta_opt, alpha_opt, lambda_opt = beta_est, alpha_est, lambda_est
        nll_opt = new_nll


#saving optimized parms
end_time = time.time()
elapsed_time = end_time-start_time

grads = grad_func(beta_opt, alpha_opt, lambda_opt)
lInf_grads = max(tf.reduce_max(tf.abs(grads[0])), 
                 tf.reduce_max(tf.abs(grads[1])), 
                 tf.reduce_max(tf.abs(grads[2])))

total_params = tf.cast(tf.size(beta) + tf.size(alpha) + tf.size(lambda_), tf.float32)

mae_error = (tf.reduce_sum(tf.math.abs(beta_opt - beta)) +
             tf.reduce_sum(tf.math.abs(alpha_opt - alpha)) +
             tf.reduce_sum(tf.math.abs(lambda_opt - lambda_))) / total_params

lInf_error = max(tf.reduce_max(tf.math.abs(beta_opt - beta)),
                 tf.reduce_max(tf.math.abs(alpha_opt - alpha)),
                 tf.reduce_max(tf.math.abs(lambda_opt - lambda_)))

errors = tf.stack((lInf_grads, mae_error, lInf_error))

os.makedirs("estData", exist_ok=True)
np.save('estData/errors_' + run + '.npy', errors.numpy())
np.save("estData/beta_opt_" + run + ".npy", beta_opt.numpy())
np.save("estData/alpha_opt_" + run + ".npy", alpha_opt.numpy())
np.save("estData/lambda_opt_" + run + ".npy", lambda_opt.numpy()) 
np.save('estData/time_' + run + '.npy', elapsed_time)

print('Finished with elapsed time (s):', elapsed_time)