# used for both the by n cand op_n case
# setwd(dirname(rstudioapi::getSourceEditorContext()$path))
args = commandArgs(trailingOnly=TRUE)

suppressMessages({
  library(doParallel)
  library(caret)
})

seed = ifelse(length(args) > 0, args[1], 42)
K = ifelse(length(args) > 1, args[2], 2)
n = ifelse(length(args) > 2, args[3], 600)
run = paste('seed=', seed, '_K=', K, '_n=', n, sep='')

output_file = paste('evaData/satMat_', run, '.csv', sep='')
if (file.exists(output_file)) {
  message("File ", output_file, " Already exists. Exiting script.")
  quit(save = "no", status = 0)
}

seed = as.numeric(seed)
K = as.numeric(K); n = as.numeric(n)

set.seed(seed)
print(run)


# --------------------------------
#        At Last Timepoint
# --------------------------------

piK = function(
  K, X, W, A, B, y
  ) {

  aK = A[,,K]; xK = X[,,K]; wK = W[,,K]; bK = B[,K]
  HK = data.frame(X=X[,,1:K], W=W[,,1:K], B=B[,1:K-1], A=A[,,1:K-1])
  colnames(HK) <- gsub("\\.+", "_", colnames(HK))

  AK_set <- unique(aK)
  aK_sparse = factor(apply(aK, 1, which.max))
  aK_combos = unique(aK_sparse)
  covars = data.frame(HK, aK = aK_sparse)

  # fit Q model
  cl <- makeCluster(detectCores() - 1); registerDoParallel(cl)
  tune_grid = expand.grid(
    mtry = seq(floor(sqrt(ncol(covars))), ncol(covars), 3), 
    min.node.size = c(5, 15, 25),
    splitrule = c('variance'))
  train_control = trainControl(
    method = "cv", number = 5, allowParallel = TRUE)
  qK_model = train(
    x = covars, y = bK, method = "ranger", num.trees = 500,
    trControl = train_control, tuneGrid = tune_grid)
  stopCluster(cl); registerDoSEQ()

  # calculate utilities and get optimal policy
  utilities = matrix(0, nrow=dim(X)[1], ncol=length(aK_combos))
  for (j in 1:length(aK_combos)) {
    aK_broad = factor(rep(aK_combos[j], dim(X)[1]), levels=c(1,2,3,4))
    if(K>1) {
      covars_j = data.frame(X=X[,,1:K], W=W[,,1:K], B=B[,1:K-1], A=A[,,1:K-1], aK = aK_broad)
      colnames(covars_j) <- gsub("\\.+", "_", colnames(covars_j))
    } else {
      covars_j = data.frame(X=X[,,1:K], W=W[,,1:K], aK = aK_broad)
      colnames(covars_j) <- gsub("\\.+", "_", colnames(covars_j))
    }
    utilities[,j] = predict(qK_model, covars_j)
  }
  qK_max = apply(utilities, 1, max)
  aK_max = diag(4)[as.double(as.character(aK_combos[apply(utilities, 1, which.max)])),]
  aK_max_label = apply(aK_max, 1, function(r) paste0("a", which(r == 1), collapse = ""))

  # prepare mapping
  piK_map = list(key = do.call(paste, c(HK, sep = "_")), val = aK_max_label, key_df = HK)

  return(list(qK_max=qK_max, aK_max = aK_max, piK_map = piK_map))
}



# --------------------------------
#        At Timepoint k, k < K
# --------------------------------

pik = function(
  k, X, W, A, B, qk_max
  ) {

  # prepare Q model 
  ak = A[,,k]
  ak_sparse = factor(apply(ak, 1, which.max))
  ak_combos = unique(ak_sparse)

  if(k>1){
    Hk = data.frame(X=X[,,1:k], W=W[,,1:k], B=B[,1:k-1], A=A[,,1:k-1])
    colnames(Hk) <- gsub("\\.+", "_", colnames(Hk))
  } else {
    Hk = data.frame(X=X[,,1:k], W=W[,,1:k])
    colnames(Hk) <- gsub("\\.+", "_", colnames(Hk))
  }
  covars = data.frame(Hk, ak = ak_sparse)

  # fit Q model
  cl <- makeCluster(detectCores() - 1); registerDoParallel(cl)
  tune_grid = expand.grid(
    mtry = seq(floor(sqrt(ncol(covars))), ncol(covars), 3), 
    min.node.size = c(5, 15, 25),
    splitrule = c('variance'))
  train_control = trainControl(
    method = "cv", number = 5, allowParallel = TRUE)
  qk_model = train(
    x = covars, y = qk_max, method = "ranger", num.trees = 500,
    trControl = train_control, tuneGrid = tune_grid)
  stopCluster(cl); registerDoSEQ()

  # calculate utilities and get optimal policy
  utilities = matrix(0, nrow=dim(X)[1], ncol=length(ak_combos))
  for (j in 1:length(ak_combos)) {
    ak_broad = factor(rep(ak_combos[j], dim(X)[1]), levels=c(1,2,3,4))
    if(k>1) {
      covars_j = data.frame(X=X[,,1:k], W=W[,,1:k], B=B[,1:k-1], A=A[,,1:k-1], ak = ak_broad)
      colnames(covars_j) <- gsub("\\.+", "_", colnames(covars_j))
    } else {
      covars_j = data.frame(X=X[,,1:k], W=W[,,1:k], ak = ak_broad)
      colnames(covars_j) <- gsub("\\.+", "_", colnames(covars_j))
    }
    utilities[,j] = predict(qk_model, covars_j)
  }
  qk_max = apply(utilities, 1, max)
  ak_max = diag(4)[as.double(as.character(ak_combos[apply(utilities, 1, which.max)])),]
  ak_max_label = apply(ak_max, 1, function(r) paste0("a", which(r == 1), collapse = ""))

  # prepare mapping
  pik_map = list(key = do.call(paste, c(Hk, sep = "_")), val = ak_max_label, key_df = Hk)

  return(list(qk_max = qk_max, ak_max = ak_max, pik_map = pik_map))
}


# --------------------------------
#         Model Evaluation
# --------------------------------

# load functions and simulated data
source('genNamespace_K.R')

# observational value
e = t(apply(V,1,softmax))
initResults = c(mean(rowSums(y*e)), mean(B[,K]), mean(y))

pi_mapList = vector('list', length = K)
# estimated DTR at timepoint K
piK_opt <- piK(K, X, W, A, B, y)
pi_mapList[[K]] = piK_opt$piK_map

# estimated DTR at timepoint K-1, ..., 1
qk_max <<- piK_opt$qK_max
for (k in (K-1):1) {
  pik_opt <- pik(k, X, W, A, B, qk_max)
  pi_mapList[[k]] = pik_opt$pik_map
  qk_max <<- pik_opt$qk_max
}

# value of DTR (on testing set), n testing set to be the same as n training
newResults = evaluateValue(pi_mapList, n, K, seed, run, alpha0, alpha1, beta0, beta1)
mat = rbind(initResults, newResults)
colnames(mat) = c('E[U]', 'E[BK]', 'E[Y]')
rownames(mat) = c('observed data', 'estimated DTR')
print(mat)

write.csv(mat, paste('evaData/satMat_', run, '.csv', sep=''))

closeAllConnections()