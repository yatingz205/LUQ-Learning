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


# --- Simulate Data from R ---
# Simulate data from R and convert to TensorFlow tensors
if not os.path.exists('simData/w1_' + run + '.npy'):
    subprocess.call(['Rscript', 'simFiles_n.R', str(seed), str(n_sim), str(n), str(lab)])

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


# --- Define Helper Functions ---

def advanced_index(tensor, indices):
    return tf.transpose(tf.gather(tf.transpose(tensor, perm=[2, 0, 1]), indices, batch_dims=1))

def tau_metric_tf(r_sim_bc, r_perm):
    def discord(a, b):
        return tf.cast(
            (r_sim_bc[:,:,a] - r_sim_bc[:,:,b]) * 
            (r_perm[:,:,a] - r_perm[:,:,b]) < 0, 
            tf.float32
        )
    tau_dists = discord(0, 1) + discord(0, 2) + discord(1, 2)
    return tau_dists

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


# --- Compute tau_dists_sim Vectorized ---
v_sim = generate_qmc_samples(n_sim, dim=2)
v_extend_sim = tf.concat([tf.ones([n_sim, 1]), v_sim], axis=1) 
v_extended = tf.concat([v_sim, tf.zeros([n_sim, 1])], axis=1)
e_sim = tf.nn.softmax(v_extended, axis=1)
np.save(make_title("v_extended_sim", run), v_extend_sim)
np.save(make_title("e_sim", run), e_sim)

eR_sim = tf.argsort(e_sim, axis=1, direction="ASCENDING", stable = True) 
r_sim = tf.argsort(eR_sim, axis=1)
all_perms = tf.constant([[1, 2, 0],[2, 1, 0],[0, 1, 2],[2, 0, 1],[0, 2, 1],[1, 0, 2]], dtype=tf.int32)
r_perm = tf.argsort(all_perms, axis=1)[tf.newaxis, :, :] 
r_sim_bc = r_sim[:, tf.newaxis, :]
tau_dists_sim = tau_metric_tf(r_sim_bc, r_perm)

  
# --- Objects for w Components ---
w1bc = tf.expand_dims(w1, axis=1)  
w2bc = tf.expand_dims(w2, axis=1) 
w1_targets_bc = tf.tile(w1bc, [1, n_sim, 1])
w2_targets_bc = tf.tile(w2bc, [1, n_sim, 1])


# --- Objects for wR Components ---
all_perms_bc = all_perms[tf.newaxis, :, :]

w1R_i = tf.cast(w1R, tf.int32)
w1R_expanded = w1R_i[:, tf.newaxis, :]
w1R_alt = tf.cast(tf.reduce_all(all_perms_bc == w1R_expanded, axis=-1), tf.float32) 

w2R_i = tf.cast(w2R, tf.int32)
w2R_expanded = w2R_i[:, tf.newaxis, :]
w2R_alt = tf.cast(tf.reduce_all(all_perms_bc == w2R_expanded, axis=-1), tf.float32) 


# --- Objects for b Components ---
u1_sim = tf.matmul(e_sim, tf.transpose(x2))    
u2_sim = tf.matmul(e_sim, tf.transpose(y))


# --- Organize Parameters ---
alpha0 = tf.concat([tf.expand_dims(alpha01, 0), tf.expand_dims(alpha02, 0)], axis=0)  
alpha_temp = tf.concat([tf.expand_dims(alpha1, 0), tf.expand_dims(alpha2, 0)], axis=0)
alpha_tensor = tf.concat([alpha0, tf.expand_dims(alpha_temp, 1)], axis=1)  # Shape: [2, 7]
beta_tensor = tf.concat([tf.expand_dims(beta01, 0), beta1, tf.expand_dims(beta02, 0), beta2], axis=0) 
lambda_tensor = tf.concat([tf.expand_dims(lambda1, 0), tf.expand_dims(lambda2, 0)], axis=0)  # Shape: [2]

alpha = tf.Variable(alpha_tensor, name="alpha")
beta = tf.Variable(beta_tensor, name="beta")
lambda_ = tf.Variable(lambda_tensor, name="lambda")



# ===============================================
# Define Negative Log-Likelihood (NLL) Function 
# ===============================================
@tf.function
def NLL_func(beta, alpha, lambda_):
    eps = tf.constant(1e-30, dtype=tf.float32)

    beta01, beta1, beta02, beta2 = beta[0, :], beta[1:3, :], beta[3, :], beta[4:, :]
    alpha01, alpha1, alpha02, alpha2 = alpha[0, :-1], alpha[0, -1], alpha[1, :-1], alpha[1, -1]
    lambda1, lambda2 = lambda_[0], lambda_[1]
    
    n = tf.shape(w1)[0]
    n_sim = tf.shape(v_extend_sim)[0]


    # ---------- w1 / w2 components ----------
    # logits: [n_sim, 12]
    logits_w1 = tf.matmul(v_extend_sim, tf.concat([tf.reshape(beta01, (1, -1)), beta1], axis=0))
    logits_w2 = tf.matmul(v_extend_sim, tf.concat([tf.reshape(beta02, (1, -1)), beta2], axis=0))

    # For each patient i, evaluate likelihood under each simulated V_j
    logits_w1_bc = logits_w1[tf.newaxis, :, :]
    logits_w2_bc = logits_w2[tf.newaxis, :, :]

    # Compute log probabilities: [n, n_sim, 12]
    logits_w1_bc = tf.broadcast_to(logits_w1_bc, tf.shape(w1_targets_bc))
    logits_w2_bc = tf.broadcast_to(logits_w2_bc, tf.shape(w2_targets_bc))
    logp_w1 = -tf.nn.sigmoid_cross_entropy_with_logits(labels=w1_targets_bc, logits=logits_w1_bc)
    logp_w2 = -tf.nn.sigmoid_cross_entropy_with_logits(labels=w2_targets_bc, logits=logits_w2_bc)

    # Sum over 12 binary outcomes, transpose to [n_sim, n]
    log_comps_w1 = tf.transpose(tf.reduce_sum(logp_w1, axis=-1), perm=[1, 0])  # [n_sim, n]
    log_comps_w2 = tf.transpose(tf.reduce_sum(logp_w2, axis=-1), perm=[1, 0])  # [n_sim, n]


    # ---------- b1 / b2 components ----------
    cum_probs1 = tf.nn.sigmoid(alpha01[:, tf.newaxis, tf.newaxis] - alpha1 * u1_sim[tf.newaxis, :, :])
    b1_probs = tf.concat(
        (cum_probs1[0, tf.newaxis, :, :],
         cum_probs1[1:, :, :] - cum_probs1[:-1, :, :],
         1.0 - cum_probs1[-1, tf.newaxis, :, :]),
        axis=0) 
    log_comps_b1 = tf.math.log(tf.maximum(advanced_index(b1_probs, b1), eps))  # [n_sim, n]

    cum_probs2 = tf.nn.sigmoid(alpha02[:, tf.newaxis, tf.newaxis] - alpha2 * u2_sim[tf.newaxis, :, :])
    b2_probs = tf.concat(
        (cum_probs2[0, tf.newaxis, :, :],
         cum_probs2[1:, :, :] - cum_probs2[:-1, :, :],
         1.0 - cum_probs2[-1, tf.newaxis, :, :]),
        axis=0) 
    log_comps_b2 = tf.math.log(tf.maximum(advanced_index(b2_probs, b2), eps))  # [n_sim, n]


    # ---------- w1R / w2R components ----------
    # log_w1R_probs, log_w2R_probs: [n_sim, 6]
    log_w1R_probs = tf.nn.log_softmax(-lambda1 * tau_dists_sim, axis=1)
    log_w2R_probs = tf.nn.log_softmax(-lambda2 * tau_dists_sim, axis=1)

    log_comps_w1R = tf.reduce_sum(log_w1R_probs[:, tf.newaxis, :] * w1R_alt[tf.newaxis, :, :], axis=-1)
    log_comps_w2R = tf.reduce_sum(log_w2R_probs[:, tf.newaxis, :] * w2R_alt[tf.newaxis, :, :], axis=-1)


    # ---------- aggregate marginal log-likelihood ----------
    log_comps = (log_comps_w1 + log_comps_w2 + log_comps_b1 + log_comps_b2 + log_comps_w1R + log_comps_w2R)
    log_lik = tf.reduce_sum(tf.reduce_logsumexp(log_comps, axis=0) - tf.math.log(tf.cast(n_sim, tf.float32)))


    # ---------- regularization on alpha and lambda ----------
    penalty = (
        tf.exp(tf.minimum(-20.0 * alpha1, 80.0)) +
        tf.exp(tf.minimum(-20.0 * alpha2, 80.0)) +
        tf.exp(tf.minimum(-20.0 * lambda1, 80.0)) +
        tf.exp(tf.minimum(-20.0 * lambda2, 80.0)) +
        tf.reduce_sum(tf.exp(tf.minimum(-20.0 * (alpha01[1:] - alpha01[:-1]), 80.0))) +
        tf.reduce_sum(tf.exp(tf.minimum(-20.0 * (alpha02[1:] - alpha02[:-1]), 80.0)))
    ) / 10.0
    
    return -log_lik / tf.cast(n, tf.float32) + penalty


# =============================
# Define Gradient Function 
# =============================
@tf.function
def grad_func(beta, alpha, lambda_):  
    with tf.GradientTape() as tape: nll = NLL_func(beta, alpha, lambda_)
    gradients = tape.gradient(nll, [beta, alpha, lambda_])
    return gradients


# =============================
# Define Diagnostics Function
# =============================
def print_diagnostics(beta_est, alpha_est, lambda_est):

    with tf.GradientTape() as tape:
        nll = NLL_func(beta_est, alpha_est, lambda_est)

    grads = tape.gradient(nll, [beta_est, alpha_est, lambda_est])
    g_beta, g_alpha, g_lambda = grads
    lInf_grads = tf.reduce_max([
        tf.reduce_max(tf.abs(g_beta)),
        tf.reduce_max(tf.abs(g_alpha)),
        tf.reduce_max(tf.abs(g_lambda)),
    ])

    total_params = tf.cast(tf.size(beta) + tf.size(alpha) + tf.size(lambda_), tf.float32)
    mae_error = (
        tf.reduce_sum(tf.abs(beta_est - beta)) +
        tf.reduce_sum(tf.abs(alpha_est - alpha)) +
        tf.reduce_sum(tf.abs(lambda_est - lambda_))) / total_params
    lInf_error = tf.reduce_max([
        tf.reduce_max(tf.abs(beta_est - beta)),
        tf.reduce_max(tf.abs(alpha_est - alpha)),
        tf.reduce_max(tf.abs(lambda_est - lambda_)),])
    print(
        "nll", round(float(nll), 6),
        "lInf_grads", round(float(lInf_grads), 6),
        "mae_error", round(float(mae_error), 6),
        "lInf_error", round(float(lInf_error), 6),
    )

def loss_lambda_only(lam):
    return NLL_func(beta, alpha, lam)  

def print_diagnostics_lambda_only(lambda_est):
    with tf.GradientTape() as tape:
        nll = NLL_func(beta, alpha, lambda_est)  
    grad_lambda = tape.gradient(nll, lambda_est) 
    print("nll", float(nll),
          "lInf_grad_lambda", float(tf.reduce_max(tf.abs(grad_lambda))),
          "lambda_est", lambda_est.numpy())



