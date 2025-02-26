# Imports
import time
import os
# Suppress TensorFlow messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  
os.environ["TF_USE_LEGACY_KERAS"] = "True"
import subprocess
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from itertools import permutations


# Enable TensorFlow device placement logging for debugging
tf.debugging.set_log_device_placement(True)

# --- Simulate Data from R ---
# Simulate data from R and convert to TensorFlow tensors
if not os.path.exists('simData/vSim_' + run + '.npy'):
    subprocess.call(['Rscript', 'simFiles_n.R', str(sn), str(seed), str(n), str(n_sim)])

# --- Load and Convert Data to GPU-Compatible Tensors ---
with tf.device('/GPU:0'):
    v_sim = tf.constant(np.load(f'simData/vSim_{run}.npy'), dtype=tf.float32)
    w1 = tf.constant(np.load(f'simData/w1_{run}.npy'), dtype=tf.float32)
    beta01 = tf.constant(np.load(f'simData/beta01_{run}.npy'), dtype=tf.float32)
    beta1 = tf.constant(np.load(f'simData/beta1_{run}.npy'), dtype=tf.float32)
    w2 = tf.constant(np.load(f'simData/w2_{run}.npy'), dtype=tf.float32)
    beta02 = tf.constant(np.load(f'simData/beta02_{run}.npy'), dtype=tf.float32)
    beta2 = tf.constant(np.load(f'simData/beta2_{run}.npy'), dtype=tf.float32)
    b1 = tf.constant(np.load(f'simData/b1_{run}.npy'), dtype=tf.int32) - 1
    alpha01 = tf.constant(np.load(f'simData/alpha01_{run}.npy'), dtype=tf.float32)
    alpha1 = tf.constant(np.load(f'simData/alpha1_{run}.npy'), dtype=tf.float32)
    b2 = tf.constant(np.load(f'simData/b2_{run}.npy'), dtype=tf.int32) - 1
    alpha02 = tf.constant(np.load(f'simData/alpha02_{run}.npy'), dtype=tf.float32)
    alpha2 = tf.constant(np.load(f'simData/alpha2_{run}.npy'), dtype=tf.float32)
    x2 = tf.constant(np.load(f'simData/x2_{run}.npy'), dtype=tf.float32)
    y = tf.constant(np.load(f'simData/y_{run}.npy'), dtype=tf.float32)
    lambda1 = tf.constant(np.load(f'simData/lambda1_{run}.npy'), dtype=tf.float32)
    lambda2 = tf.constant(np.load(f'simData/lambda2_{run}.npy'), dtype=tf.float32)
    w1R = tf.constant(np.load(f'simData/w1R_{run}.npy'), dtype=tf.float32) - 1
    w2R = tf.constant(np.load(f'simData/w2R_{run}.npy'), dtype=tf.float32) - 1


# --- Define Useful Functions ---
def softmax_with_padding(x):
    max_x = tf.reduce_max(x, axis=1, keepdims=True)
    M = tf.maximum(max_x, 0)
    exp_x = tf.exp(x - M); exp_zero = tf.exp(0 - M)  
    Z = tf.reduce_sum(exp_x, axis=1, keepdims=True) + exp_zero
    p_x = exp_x / Z; p_zero = exp_zero / Z
    return tf.concat([p_x, p_zero], axis=1)

def advanced_index(tensor, indices):
    return tf.transpose(tf.gather(tf.transpose(tensor, perm=[2, 0, 1]), indices, batch_dims=1))

tfrm = tf.reduce_mean


# --- Objects for w Components ---
with tf.device('/GPU:0'):
    w1bc = tf.expand_dims(w1, axis=1)  # Shape: [600, 1, 12]
    w2bc = tf.expand_dims(w2, axis=1)  # Shape: [600, 1, 12]


# --- Objects for b Components ---
with tf.device('/GPU:0'):
    e_sim = softmax_with_padding(v_sim)          # Shape: [1000, 3]
    print('e_sim shape:', e_sim.shape)           # (1000, 3)
    u1_sim = tf.matmul(e_sim, tf.transpose(x2))  # Shape: [1000, 600]
    print('u1_sim shape:', u1_sim.shape)         # (1000, 600)
    u2_sim = tf.matmul(e_sim, tf.transpose(y))   # Shape: [1000, 600]
    print('u1_sim shape:', u1_sim.shape)         # (1000, 600)


# --- Objects for wR Components ---
with tf.device('/GPU:0'):
    all_perms = tf.constant(list(set(permutations([0, 1, 2]))), dtype=tf.float32)  # Shape: [6, 3]
    all_perms_bc = tf.expand_dims(all_perms, axis=0)  # Shape: [1, 6, 3]
    # Compare permutations
    w1R_expanded = tf.expand_dims(w1R, axis=1)       # Shape: [600, 1, 3]
    w2R_expanded = tf.expand_dims(w2R, axis=1)       # Shape: [600, 1, 3]
    
    # Check for equality across the last dimension
    w1R_alt = tf.cast(tf.reduce_sum(tf.cast(all_perms_bc == w1R_expanded, tf.float32), axis=-1) == 3, tf.float32)  # Shape: [n_sim, 6]
    w2R_alt = tf.cast(tf.reduce_sum(tf.cast(all_perms_bc == w2R_expanded, tf.float32), axis=-1) == 3, tf.float32)  # Shape: [n_sim, 6]


# --- Compute tau_dists_sim Vectorized ---
with tf.device('/GPU:0'):
    # Compute argsort indices
    eR_sim = tf.cast(tf.math.top_k(e_sim, k=3, sorted=True).indices, tf.float32)
    
    # Reshape for broadcasting
    eR_sim_expanded = tf.expand_dims(eR_sim, axis=1)        # Shape: [1000, 1, 3]
    all_perms_expanded = tf.expand_dims(all_perms, axis=0)  # Shape: [1, 6, 3]
    
    # Compute differences between sorted e_sim
    # Assuming eR_sim has shape [1000, 3]
    diff_e1 = eR_sim_expanded[:, :, 1] - eR_sim_expanded[:, :, 0]  # Shape: [1000, 6]
    diff_e2 = eR_sim_expanded[:, :, 2] - eR_sim_expanded[:, :, 0]  # Shape: [1000, 6]
    diff_e3 = eR_sim_expanded[:, :, 2] - eR_sim_expanded[:, :, 1]  # Shape: [1000, 6]
    
    # Compute differences in permutations
    diff_p1 = all_perms_expanded[:, :, 1] - all_perms_expanded[:, :, 0]  # Shape: [1, 6]
    diff_p2 = all_perms_expanded[:, :, 2] - all_perms_expanded[:, :, 0]  # Shape: [1, 6]
    diff_p3 = all_perms_expanded[:, :, 2] - all_perms_expanded[:, :, 1]  # Shape: [1, 6]
    
    # Broadcast permutation differences to match [1000, 6]
    num_perms = all_perms.shape[0]  # 6 for permutations of 3 elements
    diff_p1_broadcasted = tf.broadcast_to(diff_p1, [n_sim, 6])  # Shape: [1000, 6]
    diff_p2_broadcasted = tf.broadcast_to(diff_p2, [n_sim, 6])  # Shape: [1000, 6]
    diff_p3_broadcasted = tf.broadcast_to(diff_p3, [n_sim, 6])  # Shape: [1000, 6]
    
    # Compute tau components
    tau_comp1 = tf.where((diff_e1 * diff_p1_broadcasted) < 0, 1.0, 0.0)
    tau_comp2 = tf.where((diff_e2 * diff_p2_broadcasted) < 0, 1.0, 0.0)
    tau_comp3 = tf.where((diff_e3 * diff_p3_broadcasted) < 0, 1.0, 0.0)
    
    # Aggregate tau distances
    tau_dists_sim_values = tau_comp1 + tau_comp2 + tau_comp3  # Shape: [1000, 6]
    
    # Assign to the variable
    tau_dists_sim = tf.convert_to_tensor(tau_dists_sim_values, dtype=tf.float32)  # Shape: [1000, 6]


# --- Define Other Objects ---
with tf.device('/GPU:0'):
    alpha0 = tf.concat([tf.expand_dims(alpha01, 0), tf.expand_dims(alpha02, 0)], axis=0)  
    alpha_temp = tf.concat([tf.expand_dims(alpha1, 0), tf.expand_dims(alpha2, 0)], axis=0)
    alpha_tensor = tf.concat([alpha0, tf.expand_dims(alpha_temp, 1)], axis=1)  # Shape: [2, 7]
    beta_tensor = tf.concat([tf.expand_dims(beta01, 0), beta1, tf.expand_dims(beta02, 0), beta2], axis=0) 
    lambda_tensor = tf.concat([tf.expand_dims(lambda1, 0), tf.expand_dims(lambda2, 0)], axis=0)  # Shape: [2]

    alpha = tf.Variable(alpha_tensor, name="alpha")
    beta = tf.Variable(beta_tensor, name="beta")
    lambda_ = tf.Variable(lambda_tensor, name="lambda")


# --- Define Negative Log-Likelihood (NLL) Function ---
@tf.function
def NLL_func(beta, alpha, lambda_):   
    #extrating variables
    beta01, beta1, beta02, beta2 = beta[0,:], beta[1:3,:], beta[3,:], beta[4:,:]
    alpha01, alpha1, alpha02, alpha2 = alpha[0,:-1], alpha[0,-1], alpha[1,:-1], alpha[1,-1]
    lambda1, lambda2 = lambda_[0], lambda_[1]

    #w1 and w2 components
    w1_probs = tf.nn.sigmoid(beta01[tf.newaxis,:] + (v_sim @ beta1))[tf.newaxis,:,:]
    inner_comps1 = w1bc*tf.math.log(w1_probs) + (1-w1bc)*tf.math.log((1-w1_probs))
    comps_w1 = tf.transpose(tf.exp(tf.reduce_sum(inner_comps1, axis=-1)))
    w2_probs = tf.nn.sigmoid(beta02[tf.newaxis,:] + (v_sim @ beta2))[tf.newaxis,:,:]
    inner_comps2 = w2bc*tf.math.log(w2_probs) + (1-w2bc)*tf.math.log((1-w2_probs))
    comps_w2 = tf.transpose(tf.exp(tf.reduce_sum(inner_comps2, axis=-1)))

    #b1 components
    cum_probs1 = tf.nn.sigmoid(alpha01[:,tf.newaxis,tf.newaxis]-alpha1*u1_sim[tf.newaxis,:,:])
    b1_probs = tf.concat((cum_probs1[0,tf.newaxis,:,:], 
                            cum_probs1[1:,:,:]-cum_probs1[:-1,:,:], 
                            1-cum_probs1[-1,tf.newaxis,:,:]), axis=0)
    comps_b1 = tf.maximum(advanced_index(b1_probs, b1), 1e-6)

    #b2 components
    cum_probs2 = tf.nn.sigmoid(alpha02[:,tf.newaxis,tf.newaxis]-alpha2*u2_sim[tf.newaxis,:,:])
    b2_probs = tf.concat((cum_probs2[0,tf.newaxis,:,:], 
                            cum_probs2[1:,:,:]-cum_probs2[:-1,:,:], 
                            1-cum_probs2[-1,tf.newaxis,:,:]), axis=0)
    comps_b2 = tf.maximum(advanced_index(b2_probs, b2), 1e-6)

    #w1R and w2R components
    exp_term1 = tf.exp(-lambda1 * tau_dists_sim)
    w1R_probs = exp_term1 / tf.reduce_sum(exp_term1, axis=0)
    exp_term2 = tf.exp(-lambda2 * tau_dists_sim)
    w2R_probs = exp_term2 / tf.reduce_sum(exp_term2, axis=0)
    comps_w1R = tf.reduce_sum(w1R_probs[:,tf.newaxis,:] * w1R_alt[tf.newaxis,:,:], axis=-1)
    comps_w2R = tf.reduce_sum(w2R_probs[:,tf.newaxis,:] * w2R_alt[tf.newaxis,:,:], axis=-1)

    #aggregation + log-prior + penalty
    comps = comps_w1 * comps_w2 * comps_b1 * comps_b2 * comps_w1R * comps_w2R
    log_lik = tf.reduce_sum(tf.math.log(tf.reduce_mean(comps, axis=0)))
    penalty = (
        tf.exp(-100 * alpha1) +
        tf.exp(-100 * alpha2) +
        tf.exp(-100 * lambda1) +
        tf.exp(-100 * lambda2) +
        tf.reduce_sum(tf.exp(-100 * (alpha01[1:] - alpha01[:-1]))) +
        tf.reduce_sum(tf.exp(-100 * (alpha02[1:] - alpha02[:-1])))) / 100.0
    return -log_lik/n + penalty


# --- Define Gradient Function ---
@tf.function
def grad_func(beta, alpha, lambda_):  
    with tf.GradientTape() as tape: nll = NLL_func(beta, alpha, lambda_)
    gradients = tape.gradient(nll, [beta, alpha, lambda_])
    return gradients


# --- Define Diagnostics Function ---
def print_diagnostics(beta_est, alpha_est, lambda_est):
    nll = NLL_func(beta_est, alpha_est, lambda_est)
    grads = grad_func(beta_est, alpha_est, lambda_est)
    lInf_grads = max(tf.reduce_max(grads[0]), tf.reduce_max(grads[1]), tf.reduce_max(grads[2]))
    total_params = tf.cast(tf.size(beta) + tf.size(alpha) + tf.size(lambda_), tf.float32)
    
    l1_error = (tf.reduce_sum(tf.math.abs(beta_est-beta)) +
                tf.reduce_sum(tf.math.abs(alpha_est-alpha)) +
                tf.reduce_sum(tf.math.abs(lambda_est-lambda_)))/total_params
    lInf_error = max(tf.reduce_max(tf.math.abs(beta_est-beta)),
                     tf.reduce_max(tf.math.abs(alpha_est-alpha)),
                     tf.reduce_max(tf.math.abs(lambda_est-lambda_)))
    print('nll', round(nll.numpy().tolist(), 4), 
          'lInf_grads', round(lInf_grads.numpy().tolist(), 6), 
          'l1_error', round(l1_error.numpy().tolist(), 6), 
          'lInf_error', round(lInf_error.numpy().tolist(), 6))

