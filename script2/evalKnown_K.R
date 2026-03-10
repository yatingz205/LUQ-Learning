args = commandArgs(trailingOnly=TRUE)

suppressMessages({
  library(doParallel)
  library(reticulate)
  library(caret)
})

seed = ifelse(length(args) > 0, args[1], 42)
K = ifelse(length(args) > 1, args[2], 2)
n = ifelse(length(args) > 2, args[3], 600)
run = paste('seed=', seed, '_K=', K, '_n=', n, sep='')

output_file = paste('evaData/knownMat_', run, '.csv', sep='')
if (file.exists(output_file)) {
  message("File ", output_file, " Already exists. Exiting script.")
  quit(save = "no", status = 0)
}

seed = as.numeric(seed)
dir.create("evaData", showWarnings = FALSE, recursive = TRUE)
K = as.numeric(K); n = as.numeric(n)

set.seed(seed)
print(run)


use_condaenv("luql_env", required = TRUE)
np = import("numpy")
allPerms = matrix(c(1,2,0,2,1,0,0,1,2,2,0,1,0,2,1,1,0,2), ncol=3, byrow=T)+1
source('helperFuncs.R')

#load training data
makeTitle = function(objName) paste('simData/', objName, '_', run, '.npy', sep='')
V = np$load(makeTitle('V')); W = np$load(makeTitle('W'))
A = np$load(makeTitle('A')); X = np$load(makeTitle('X'))
y = np$load(makeTitle('y')); B = np$load(makeTitle('B'))
eSim = np$load(makeTitle('e_sim')); vextendSim = np$load(makeTitle('v_extended_sim'));
beta0 = np$load(makeTitle('beta0')); beta1 = np$load(makeTitle('beta1'))
alpha0 = np$load(makeTitle('alpha0')); alpha1 = np$load(makeTitle('alpha1'))
gamma0 = np$load(makeTitle('gamma0')); gamma1 = np$load(makeTitle('gamma1'))
shift_x = np$load(makeTitle('shift_x')); scale_x = np$load(makeTitle('scale_x'))

#load estimates
makeTitle = function(objName) paste('estData/', objName, '_', run, '.npy', sep='')
beta_opt = np$load(makeTitle('beta_opt'))
alpha_opt = np$load(makeTitle('alpha_opt'))
alpha0_opt = alpha_opt[-nrow(alpha_opt), ]; alpha1_opt = alpha_opt[nrow(alpha_opt), ]
beta0_opt = beta_opt[, , 1]; beta1_opt = beta_opt[, , -1]


# --------------------------------
#        At Last Timepoint
# --------------------------------

piK = function(
  K, X, A, V, y, 
  alpha0_opt, alpha1_opt, beta0_opt, beta1_opt
) {

  E = t(apply(V,1,softmax))
  aK = A[,,K]; xK = X[,,K]; 
  aK_label <- paste0('a', factor(apply(aK, 1, which.max), levels=c(1,2,3,4)))
  AK_set <- paste0("a", 1:4)

  # Fit conditional Y models
  res = cond_y(y, xK, aK_label, model = 'rf')
  mean_y = res$mean_y; y_models = res$y_models; design_names = res$design_names

  # Get conditional E
  mean_e = E
  
  # Get utility U
  utilities = matrix(0, nrow=nrow(xK), ncol=length(AK_set))
  for (j in 1:length(AK_set)) {
    aK_label_dum = rep(AK_set[j], nrow(xK))
    mean_y_pred = cond_y_predict(xK = xK, aK = aK_label_dum, y_models, type = 'rf')
    utilities[,j] = rowSums(mean_y_pred * mean_e)
  }
  qK_max <- apply(utilities, 1, max)

  return(list(qK_max=qK_max, y_model = y_models))
}


# --------------------------------
#        At Timepoint k, k < K
# --------------------------------

pik <- function(k, X, A, V, qk_max) {

  E = t(apply(V,1,softmax))
  Ak_set <- paste0("a", 1:4)
  A_label <- array(NA_character_, dim = c(dim(A)[1],k)) # dim = (n, k)
  for(l in 1:k) A_label[, l] <- paste0("a", apply(A[,,l,drop = FALSE], 3, max.col))
  
  if(k == 1) {
    Hk_df <- data.frame(Xk = X[,,1:k], E = E)
  } else {
    Hk_df <- data.frame(
      Xk = X[,,1:k],
      A_hist_label = A_label[,1:(k-1)], E = E)
    colnames(Hk_df)
  }
  covars <- data.frame(Hk_df, Ak = A_label[,k], check.names = FALSE)
  qk_model <- fit_rf(x_df = covars, y = qk_max)

  utilities <- matrix(0, nrow = dim(X)[1], ncol = length(Ak_set))
  for (j in 1:length(Ak_set)) {
    aK_label_dum = rep(Ak_set[j], nrow(Hk_df))
    covars_j <- data.frame(Hk_df, Ak = aK_label_dum, check.names = FALSE)
    utilities[, j] <- predict(qk_model, newdata = covars_j)
  }
  qk_max <- apply(utilities, 1, max)

  return(list(qk_max = qk_max, y_model = qk_model))
}


# --------------------------------
#         Model Evaluation
# --------------------------------

# load functions and simulated data
source('genNamespace_K.R')

# observational value
e = t(apply(V,1,softmax))
initResults = c(mean(rowSums(y*e)), mean(B[,K]), mean(y))

q_model_list = vector('list', length = K)
# estimated DTR at timepoint K
piK_opt <- piK(
  K, X, A, V, y, 
  alpha0_opt, alpha1_opt, beta0_opt, beta1_opt)
q_model_list[[K]] <- piK_opt$y_model

# estimated DTR at timepoint K-1, ..., 1
qk_max <<- piK_opt$qK_max
for (k in (K-1):1) {
  pik_opt <- pik(k, X, A, V, qk_max)
  q_model_list[[k]] <- pik_opt$y_model
  qk_max <<- pik_opt$qk_max
}

# value of DTR (on testing set), n testing set to be the same as n training
newResults = evaluateValue(
  q_model_list = q_model_list,
  n_test = n,
  test_seed = seed + 1000,
  K = K,
  alpha0 = alpha0, alpha1 = alpha1,
  beta0 = beta0, beta1 = beta1,
  gamma0 = gamma0, gamma1 = gamma1,
  alpha0_opt = alpha0_opt, alpha1_opt = alpha1_opt,
  beta0_opt = beta0_opt, beta1_opt = beta1_opt,
  vextendSim = vextendSim,
  eSim = eSim,
  type = "known"
)

mat = rbind(initResults, newResults)
colnames(mat) = c('E[U]', 'E[BK]', 'E[Y]')
rownames(mat) = c('observed data', 'estimated DTR')
print(mat)

write.csv(mat, paste('evaData/knownMat_', run, '.csv', sep=''))

closeAllConnections()
