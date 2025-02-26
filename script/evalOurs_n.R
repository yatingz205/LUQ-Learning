# used for both the by n cand op_n case
# setwd(dirname(rstudioapi::getSourceEditorContext()$path))
args = commandArgs(trailingOnly=TRUE)

suppressMessages({
  library(doParallel)
  library(caret)
})

sn = args[1]
seed = args[2]
n = args[3]
run = paste('sn=', sn, '_seed=', seed, '_n=', n, sep='')

output_file = paste('evaData/oursMat_', run, '.csv', sep='')
if (file.exists(output_file)) {
  message("File ", output_file, " Already exists. Exiting script.")
  quit(save = "no", status = 0)
}

seed = as.numeric(seed)
n = as.numeric(n)

set.seed(seed)
print(run)


# --------------------------------
#        At Last Timepoint
# --------------------------------

piK = function(
  x1, x2, a1, a2, b1, b2, w1, w2, w1R, w2R, vSim, y, 
  alpha01_opt, alpha1_opt, alpha02_opt, alpha2_opt,
  beta01_opt, beta1_opt, beta02_opt, beta2_opt, 
  lambda1_opt, lambda2_opt
  ) {

  A2_set <- unique(a2)

  # Fit conditional Y models
  colnames(a2) <- paste0("a", 1:ncol(a2))
  res = cond_y(y, x2, a2, model = 'rf')
  mean_y = res$mean_y; y_models = res$y_models; design_names = res$design_names

  # Get conditional E
  mean_e = cond_exp(
    x1, x2, a1, a2, b1, b2, w1, w2, w1R, w2R, vSim,
    alpha01_opt, alpha1_opt, alpha02_opt, alpha2_opt,
    beta01_opt, beta1_opt, beta02_opt, beta2_opt, 
    lambda1_opt, lambda2_opt)
  
  # Get utility U
  utilities = matrix(nrow=nrow(x2), ncol=nrow(A2_set))
  for (j in 1:nrow(A2_set)) {
    a2_broad = matrix(rep(A2_set[j,], nrow(x2)), nrow=nrow(x2), byrow=T)
    mean_y_pred = cond_y_predict(x2, a2_broad, y_models, design_names, type = 'rf')
    utilities[,j] = rowSums(mean_y_pred * mean_e)
  }

  # Take max to obtain policy
  q2_max = apply(utilities, 1, max)
  a2_max = A2_set[max.col(utilities, ties.method = "first"), ,drop = FALSE]

  a1_label = apply(a1, 1, function(row) {
    active_indices <- which(row == 1)
    paste0('a', active_indices, collapse = "")
  })
  a2_max_label = apply(a2_max, 1, function(row) {
    active_indices <- which(row == 1)
    paste0('a', active_indices, collapse = "")
  })
  colnames(w1) = paste('W1', 1:ncol(w1), sep = '')
  colnames(w2) = paste('W2', 1:ncol(w2), sep = '')
  w1R_sparse = factor(rowSums(t(t(w1R) * c(20,5,1))), labels=seq(1,6))
  w2R_sparse = factor(rowSums(t(t(w2R) * c(20,5,1))), labels=seq(1,6))
  key_df = data.frame(x1, x2, a1_label = a1_label, b1 = b1, w1, w2, W1R = w1R_sparse, W2R = w2R_sparse)
  pi2_map = list(key = do.call(paste, c(key_df, sep = "_")), val = a2_max_label, key_df = key_df)

  return(list(q2_max=q2_max, a2_max = a2_max, pi2_map = pi2_map))
}


# --------------------------------
#        At Timepoint k, k < K
# --------------------------------

pik = function(
  x1, a1, w1, w1R, w2R, q2_max
  ) {

  a1_sparse = factor(apply(a1, 1, which.max))
  a1_combos = unique(a1_sparse)
  colnames(w1) = paste0('W1', 1:ncol(w1))
  w1R_sparse = factor(rowSums(t(t(w1R) * c(20,5,1))), labels=seq(1,6))
  covars = data.frame(x1, w1, w1R_sparse, a1 = a1_sparse)

  cl <- makeCluster(detectCores() - 1); registerDoParallel(cl)
  tune_grid = expand.grid(
    mtry = seq(floor(sqrt(ncol(covars))), ncol(covars), 3), 
    min.node.size = c(5, 15, 25),
    splitrule = c('variance'))
  train_control = trainControl(
    method = "cv", number = 5, allowParallel = TRUE)
  q1_model = train(
    x = covars, y = q2_max, method = "ranger", num.trees = 500,
    trControl = train_control, tuneGrid = tune_grid)
  stopCluster(cl); registerDoSEQ()

  utilities = matrix(nrow=nrow(x1), ncol=length(a1_combos))
  for (j in 1:length(a1_combos)) {
    a1_broad = factor(rep(a1_combos[j], nrow(x1)), levels=c(1,2,3,4))
    covars_j = data.frame(x1, w1, w1R_sparse, a1 = a1_broad)
    colnames(covars_j) = colnames(covars)
    utilities[,j] = predict(q1_model, covars_j)
  }
  q1_max = apply(utilities, 1, max)
  a1_max = diag(4)[as.double(as.character(a1_combos[apply(utilities, 1, which.max)])),]
  a1_max_label = apply(a1_max, 1, function(r) paste0("a", which(r == 1), collapse = ""))

  colnames(w1) = paste('W1', 1:ncol(w1), sep = '')
  key_df = data.frame(x1, w1, W1R = w1R_sparse)
  pi1_map = list(key = do.call(paste, c(key_df, sep = "_")), val = a1_max_label, key_df = key_df)

  return(list(q1_max = q1_max, a1_max = a1_max, pi1_map = pi1_map))
}


# --------------------------------
#         Model Evaluation
# --------------------------------

# load functions and simulated data
source('genNamespace_n.R')

# observational value
e = t(apply(v,1,softmax))
initResults = c(mean(rowSums(y*e)), mean(b2), mean(y))

# estimated DTR at timepoint 2
pi2_opt <- piK(
  x1, x2, a1, a2, b1, b2, w1, w2, w1R, w2R, vSim, y, 
  alpha01_opt, alpha1_opt, alpha02_opt, alpha2_opt,
  beta01_opt, beta1_opt, beta02_opt, beta2_opt, 
  lambda1_opt, lambda2_opt)
pi2_map = pi2_opt$pi2_map

# estimated DTR at timepoint 1
pi1_opt <- pik(x1, a1, w1, w1R, w2R, q2_max = pi2_opt$q2_max)
pi1_map = pi1_opt$pi1_map

# value of DTR (on testing set), n testing set to be the same as n training
newResults = evaluateValue(
  pi1_map, pi2_map, 
  n=nrow(x1), seed=seed, run=run, 
  alpha01, alpha1, alpha02, alpha2, 
  beta01, beta1, beta02, beta2, 
  lambda1, lambda2, gamma01, gamma11, gamma02, gamma12)
mat = rbind(initResults, newResults)
colnames(mat) = c('E[U]', 'E[B2]', 'E[Y]')
rownames(mat) = c('observed data', 'estimated DTR')
print(mat)

write.csv(mat, paste('evaData/oursMat_', run, '.csv', sep=''))


