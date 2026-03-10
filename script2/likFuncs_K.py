#imports
import time
import os
import sys
# Suppress TensorFlow messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  
os.environ["TF_USE_LEGACY_KERAS"] = "True" 
import subprocess
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from itertools import permutations


# --- Simulate Data from R ---
# Simulate data from R and convert to TensorFlow tensors
if not os.path.exists('simData/W_' + run + '.npy'):
    subprocess.call(['Rscript', 'simFiles_K.R', str(seed), str(n_sim), str(n), str(K)])

W = tf.constant(np.load('simData/W_' + run + '.npy'), dtype=tf.float32)
beta0 = tf.constant(np.load('simData/beta0_' + run + '.npy'), dtype=tf.float32)
beta1 = tf.constant(np.load('simData/beta1_' + run + '.npy'), dtype=tf.float32)
B = tf.constant(np.load('simData/B_' + run + '.npy'), dtype=tf.int64) - 1
alpha0 = tf.constant(np.load('simData/alpha0_' + run + '.npy'), dtype=tf.float32)
alpha1 = tf.constant(np.load('simData/alpha1_' + run + '.npy'), dtype=tf.float32)
X = tf.constant(np.load('simData/X_' + run + '.npy'), dtype=tf.float32)
y = tf.constant(np.load('simData/y_' + run + '.npy'), dtype=tf.float32)

beta = tf.Variable(tf.concat((beta0[:, :, tf.newaxis], beta1), axis=-1))
alpha = tf.Variable(tf.concat((alpha0, alpha1[tf.newaxis, :]), axis=0))


def generate_qmc_samples(n_sim, dim=2, seed=seed):
    halton_uniform = tfp.mcmc.sample_halton_sequence(
        dim=dim,
        num_results=n_sim,
        randomized=True,
        seed=seed
    )
    v_sim = tfp.distributions.Normal(0., 1.).quantile(halton_uniform)
    return v_sim

def make_title(obj_name, run):
    return f"simData/{obj_name}_{run}.npy"

v_sim = generate_qmc_samples(n_sim, dim=2)
v_extend_sim = tf.concat([tf.ones([n_sim, 1]), v_sim], axis=1) 
v_extended = tf.concat([v_sim, tf.zeros([n_sim, 1])], axis=1)
e_sim = tf.nn.softmax(v_extended, axis=1)
np.save(make_title("v_extended_sim", run), v_extend_sim)
np.save(make_title("e_sim", run), e_sim)

W_b = W[tf.newaxis, :, :, :]
u_sim =  tf.reduce_sum(e_sim[:, None, :, None] * X[None, :, :, 1:], axis=2)



# ===============================================
# Define Negative Log-Likelihood (NLL) Function 
# ===============================================
@tf.function
def NLL_func(beta, alpha):
    eps = tf.constant(1e-30, dtype=tf.float32)
    
    beta0 = beta[tf.newaxis, :, :, 0]
    beta1 = tf.transpose(beta[:, :, 1:], perm=[2, 0, 1])
    alpha1 = alpha[-1]; alpha0 = alpha[:-1]
    
    n = tf.shape(W_b)[1]
    n_sim = tf.shape(v_sim)[0]
    
    
    # ---------- W components ----------
    logits_w = beta0 + tf.tensordot(v_sim, beta1, axes=[[1], [0]])
    logits_w_bc = logits_w[:, tf.newaxis, :, :]
    
    # ---------- Process in batches over patients ----------
    n_batches = (n + batch_size - 1) // batch_size
    
    def body(i, total_log_lik):
        start = i * batch_size
        end = tf.minimum(start + batch_size, n)
        
        # Slice batch data
        W_batch = W_b[:, start:end, :, :]
        u_batch = u_sim[:, start:end, :]
        B_batch = B[start:end, :]
        
        # W likelihood for batch
        target_shape = [n_sim, end-start, tf.shape(W_b)[2], tf.shape(W_b)[3]]
        logits_batch = tf.broadcast_to(logits_w_bc, target_shape)
        w_targets_batch = tf.broadcast_to(W_batch, target_shape)
        logp_w = -tf.nn.sigmoid_cross_entropy_with_logits(labels=w_targets_batch, logits=logits_batch)
        log_comps_w = tf.reduce_sum(logp_w, axis=[2, 3])
        
        # B likelihood for batch
        cum_probs = tf.nn.sigmoid(
            alpha0[:, tf.newaxis, tf.newaxis, :] - 
            alpha1[tf.newaxis, tf.newaxis, tf.newaxis, :] * u_batch[tf.newaxis, :, :, :]
        )
        cum_probs = tf.transpose(cum_probs, perm=[1, 2, 3, 0])
        b_probs = tf.concat((
            cum_probs[..., :1],
            cum_probs[..., 1:] - cum_probs[..., :-1],
            1.0 - cum_probs[..., -1:]
        ), axis=-1)
        
        B_expanded = tf.expand_dims(B_batch, axis=0)
        B_tiled = tf.tile(B_expanded, [n_sim, 1, 1])
        b_probs_gathered = tf.gather(b_probs, B_tiled, axis=-1, batch_dims=3)
        log_comps_b = tf.reduce_sum(tf.math.log(tf.maximum(b_probs_gathered, eps)), axis=-1)
        
        # Aggregate batch
        log_comps = log_comps_w + log_comps_b
        batch_log_lik = tf.reduce_sum(tf.reduce_logsumexp(log_comps, axis=0) - tf.math.log(tf.cast(n_sim, tf.float32)))
        
        return i + 1, total_log_lik + batch_log_lik
    
    _, total_log_lik = tf.while_loop(
        lambda i, _: i < n_batches,
        body,
        [tf.constant(0), tf.constant(0.0, dtype=tf.float32)]
    )
    
    # ---------- Regularization ----------
    penalty = (
    tf.reduce_sum(tf.exp(tf.minimum(-50.0 * alpha1, 80.0))) +
    tf.reduce_sum(tf.exp(tf.minimum(-50.0 * (alpha0[1:, :] - alpha0[:-1, :]), 80.0)))
    ) / 10.0
    
    tf.debugging.assert_all_finite(total_log_lik, "total_log_lik non-finite")
    tf.debugging.assert_all_finite(penalty, "penalty non-finite")

    return -total_log_lik / tf.cast(n, tf.float32) + penalty


# =============================
# Define Gradient Function 
# =============================
@tf.function
def grad_func(beta, alpha):
    with tf.GradientTape() as tape: nll = NLL_func(beta, alpha)
    gradients = tape.gradient(nll, [beta, alpha])
    return gradients


# =============================
# Define Diagnostics Function
# =============================
def print_diagnostics(beta_est, alpha_est):
    theta_dim = tf.cast(tf.reduce_prod(beta_est.shape) + tf.reduce_prod(alpha_est.shape), tf.float32)
    nll = NLL_func(beta_est, alpha_est)
    grads = grad_func(beta_est, alpha_est)
    lInf_grads = max(tf.reduce_max(grads[0]), tf.reduce_max(grads[1]))
    l1_grads = (tf.reduce_sum(tf.abs(grads[0])) + tf.reduce_sum(tf.abs(grads[1]))) / theta_dim
    mae_error = (tf.reduce_sum(tf.abs(beta_est - beta)) 
                + tf.reduce_sum(tf.abs(alpha_est - alpha))) / theta_dim
    lInf_error = max(tf.reduce_max(tf.abs(beta_est - beta)), 
                     tf.reduce_max(tf.abs(alpha_est - alpha)))
    print('nll', round(nll.numpy().tolist(), 4),
          'l1_grads', round(l1_grads.numpy().tolist(), 6),
          'lInf_grads', round(lInf_grads.numpy().tolist(), 6),
          'mae_error', round(mae_error.numpy().tolist(), 6),
          'lInf_error', round(lInf_error.numpy().tolist(), 6))

