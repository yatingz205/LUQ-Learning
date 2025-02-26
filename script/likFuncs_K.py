#imports
import time
import os
import sys
import subprocess
# Suppress TensorFlow messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  
os.environ["TF_USE_LEGACY_KERAS"] = "True" 
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from itertools import permutations


#simulate data from R and convert to tf.python
if not os.path.exists('simData/vSim_' + run + '.npy'):
    subprocess.call(['Rscript', 'simFiles_K.R', str(seed), str(K), str(n), str(n_sim)])

# --- Load and Convert Data to GPU-Compatible Tensors ---
with tf.device('/GPU:0'):
    v_sim = tf.constant(np.load('simData/vSim_' + run + '.npy'), dtype=tf.float32)
    W = tf.constant(np.load('simData/W_' + run + '.npy'), dtype=tf.float32)
    beta0 = tf.constant(np.load('simData/beta0_' + run + '.npy'), dtype=tf.float32)
    beta1 = tf.constant(np.load('simData/beta1_' + run + '.npy'), dtype=tf.float32)
    B = tf.constant(np.load('simData/B_' + run + '.npy'), dtype=tf.int64) - 1
    alpha0 = tf.constant(np.load('simData/alpha0_' + run + '.npy'), dtype=tf.float32)
    alpha1 = tf.constant(np.load('simData/alpha1_' + run + '.npy'), dtype=tf.float32)
    X = tf.constant(np.load('simData/X_' + run + '.npy'), dtype=tf.float32)
    y = tf.constant(np.load('simData/y_' + run + '.npy'), dtype=tf.float32)

# Variables
with tf.device('/GPU:0'):
    beta = tf.Variable(tf.concat((beta0[:, :, tf.newaxis], beta1), axis=-1))
    alpha = tf.Variable(tf.concat((alpha0, alpha1[tf.newaxis, :]), axis=0))

# Extra objects
softmax = lambda x: tf.nn.softmax(tf.concat([x, tf.zeros((x.shape[0], 1), dtype=tf.float32)], 1))
W_b = W[:, tf.newaxis, :, :]
e_sim = softmax(v_sim)
u_sim = tf.reduce_sum(e_sim[tf.newaxis, ..., tf.newaxis] * X[:, tf.newaxis, :, 1:], axis=2)
u_sim = tf.transpose(u_sim, perm=[0, 2, 1])[:, :, tf.newaxis, :]


# --- Define Negative Log-Likelihood (NLL) Function ---
@tf.function
def NLL_func(beta, alpha, tol=1e-32):
    # w components
    beta0_b = beta[tf.newaxis, :, :, 0]
    beta1_b = tf.transpose(beta[:, :, 1:], perm=[2, 0, 1])
    w_comps = beta0_b + tf.tensordot(v_sim, beta1_b, axes=[[1], [0]])
    w_comps = tf.nn.sigmoid(w_comps)[tf.newaxis, ...]
    w_comps = tf.reduce_prod(w_comps + (1 - W_b) * (1 - 2 * w_comps), axis=[2, 3])
    
    # b components
    alpha1_b = tf.reshape(alpha[-1], (1, K, 1, 1))
    alpha0_b = tf.transpose(alpha[:-1])[tf.newaxis, ..., tf.newaxis]
    b_comps = tf.nn.sigmoid(alpha0_b - alpha1_b * u_sim)
    b_comps = tf.concat((
        b_comps[:, :, :1, :], 
        b_comps[:, :, 1:, :] - b_comps[:, :, :-1, :], 
        1 - b_comps[:, :, -1:, :]
    ), axis=2)

    indices = tf.expand_dims(B, axis=-1)
    b_comps = tf.gather_nd(b_comps, indices, batch_dims=2)
    b_comps = tf.reduce_prod(b_comps, axis=1)
    
    # Aggregation + log-prior + penalty
    comps = w_comps * b_comps
    scaled_loglik = tf.reduce_mean(tf.math.log(tf.maximum(tf.reduce_mean(comps, axis=-1), tol)))
    constraints = tf.concat((alpha0[1:-1] - alpha0[:-2], alpha[-1:, :]), axis=0)
    penalty = tf.reduce_sum(tf.exp(-100 * constraints)) / 100
    return -scaled_loglik + penalty


# --- Define Gradient Function ---
@tf.function
def grad_func(beta, alpha):
    with tf.GradientTape() as tape: nll = NLL_func(beta, alpha)
    gradients = tape.gradient(nll, [beta, alpha])
    return gradients


# --- Define Diagnostics Function ---
def print_diagnostics(beta_est, alpha_est):
    theta_dim = tf.cast(tf.reduce_prod(beta_est.shape) + tf.reduce_prod(alpha_est.shape), tf.float32)
    nll = NLL_func(beta_est, alpha_est)
    grads = grad_func(beta_est, alpha_est)
    lInf_grads = max(tf.reduce_max(grads[0]), tf.reduce_max(grads[1]))
    l1_grads = (tf.reduce_sum(tf.abs(grads[0])) + tf.reduce_sum(tf.abs(grads[1]))) / theta_dim
    l1_error = (tf.reduce_sum(tf.abs(beta_est - beta)) + tf.reduce_sum(tf.abs(alpha_est - alpha))) / theta_dim
    lInf_error = max(tf.reduce_max(tf.abs(beta_est - beta)), tf.reduce_max(tf.abs(alpha_est - alpha)))
    print('nll', round(nll.numpy().tolist(), 4),
          'l1_grads', round(l1_grads.numpy().tolist(), 6),
          'lInf_grads', round(lInf_grads.numpy().tolist(), 6),
          'l1_error', round(l1_error.numpy().tolist(), 6),
          'lInf_error', round(lInf_error.numpy().tolist(), 6))

