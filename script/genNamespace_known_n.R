#packages and setup
suppressMessages({
  library(doParallel)
  library(reticulate)
  library(ranger)
  library(caret)
  library(aod)
})

use_condaenv("prl_env", required = TRUE)

np = import("numpy")
allPerms = matrix(c(1,2,0,2,1,0,0,1,2,2,0,1,0,2,1,1,0,2), ncol=3, byrow=T)+1
source('helperFuncs.R')
source("setGrid.R")

#load numpy files
makeTitle = function(objName) paste('simData/', objName, '_', run, '.npy', sep='')
v_extend = np$load(makeTitle('v_extend')); vextendSim = np$load(makeTitle('vextendSim'))
e = np$load(makeTitle('e')); eSim = np$load(makeTitle('eSim'))
x1 = np$load(makeTitle('x1'))
w1 = np$load(makeTitle('w1')); w2 = np$load(makeTitle('w2'))
w1R = np$load(makeTitle('w1R')); w2R = np$load(makeTitle('w2R'))
a1 = np$load(makeTitle('a1')); a2 = np$load(makeTitle('a2'))
x2 = np$load(makeTitle('x2')); y = np$load(makeTitle('y'))
b1 = np$load(makeTitle('b1')); b2 = np$load(makeTitle('b2'))
beta01 = as.vector(np$load(makeTitle('beta01'))); beta02 = as.vector(np$load(makeTitle('beta02')))
beta1 = np$load(makeTitle('beta1')); beta2 = np$load(makeTitle('beta2'))
alpha01 = as.vector(np$load(makeTitle('alpha01'))); alpha02 = as.vector(np$load(makeTitle('alpha02')))
alpha1 = as.vector(np$load(makeTitle('alpha1'))); alpha2 = as.vector(np$load(makeTitle('alpha2')))
lambda1 = as.vector(np$load(makeTitle('lambda1'))); lambda2 = as.vector(np$load(makeTitle('lambda2')))
gamma01 = np$load(makeTitle('gamma01')); gamma11 = np$load(makeTitle('gamma11'))
gamma02 = np$load(makeTitle('gamma02')); gamma12 = np$load(makeTitle('gamma12'))
shift_x2 = np$load(makeTitle('shift_x2')); scale_x2 = np$load(makeTitle('scale_x2'))

#load estimates
makeTitle = function(objName) paste('estData/', objName, '_', run, '.npy', sep='')
beta_opt = np$load(makeTitle('beta_opt'))
alpha_opt = np$load(makeTitle('alpha_opt'))
lambda_opt = np$load(makeTitle('lambda_opt'))
beta01_opt = beta_opt[1,]; beta1_opt = beta_opt[2:3,]; beta02_opt = beta_opt[4,]; beta2_opt = beta_opt[5:6,]
alpha01_opt = alpha_opt[1,1:6]; alpha1_opt = alpha_opt[1,7]; alpha02_opt = alpha_opt[2,1:6]; alpha2_opt = alpha_opt[2,7]
lambda1_opt = lambda_opt[1]; lambda2_opt = lambda_opt[2]


# fit Y models
cond_y = function(y, x2, a2, model = 'rf'){

  mean_y <- array(0, dim = c(nrow(y), ncol(y)))
  y_models <- list()

  if(model == 'betabinom'){
    for (j in 1:ncol(y)) {
      yj <- y[, j]; x2j <- x2[, j]
      design_mat = data.frame(cbind(yj, model.matrix(~-1+x2j*a2)))
      y_models[[j]] = betabin(cbind(yj, 10-yj) ~ ., ~ 1, data=design_mat,  control = list(maxit = 10000))
      mean_y[,j] = 10*predict(y_models[[j]], design_mat[,-c(1)])
    }
  } else if(model == 'rf') {
    for(j in 1:ncol(y)) {
      yj <- (y[, j]); x2j <- x2[, j]
      a2_label = apply(a2, 1, function(row) paste0(colnames(a2)[which(row == 1)], collapse = ""))
      design_mat <- data.frame(yj = yj, x2j = x2j, a2 = a2)

      cl <- makeCluster(detectCores() - 1); registerDoParallel(cl)
      config <- set_getcondY(design_mat)
      fit <- train(
        as.formula("yj ~ ."),
        data = design_mat,
        method = "ranger",
        trControl = config$tune_control,
        tuneGrid = config$tune_grid,
        num.trees = 500,
        importance = 'impurity'
      )
      stopCluster(cl); registerDoSEQ()

      y_models[[j]] <- fit
      mean_y[, j] <- predict(fit, newdata = design_mat[, -1])
    }

  }
  return(list(mean_y = mean_y, y_models = y_models, design_names = colnames(design_mat)[-1]))
}

# predict using fitted Y models
cond_y_predict = function(x2, a2, y_models, design_names, type = 'rf') {
  mean_y = array(0, dim=c(nrow(x2),3))
  if(type == 'betabinom') {
    for (j in 1:3) {
    design = data.frame(model.matrix(~-1+x2[,j]*a2))
    colnames(design) = design_names
    mean_y[,j] = 10 * predict(y_models[[j]], design, type = 'response')
    }
  } else if(type == 'rf') {
    for (j in 1:3) {
      design = data.frame(x2j = x2[,j], a2 = a2)
      colnames(design) = design_names
      mean_y[,j] = predict(y_models[[j]], design)
    }
  }
  return(mean_y)
}


# function for obtaining mapping pi2_opt
pi2_fnc = function(
  pi2_map,
  x1_new, x2_new, a1_label, e_new){

  key_new_df = data.frame(x1_new, x2_new, a1_label = a1_label, e = e_new)
  key_new = do.call(paste, c(key_new_df, sep = "_"))

  key_df = pi2_map$key_df
  val_df = data.frame(a = as.factor(pi2_map$val))

  level = levels(val_df$a)
  design = cbind(key_df, val_df)

  num_cores <- detectCores() - 1 
  cl <- makeCluster(num_cores); registerDoParallel(cl)
  config <- set_getpi(key_df)
  pi2_model <- suppressWarnings(train(
    a ~ .,
    data = design,
    method = "ranger",
    trControl = config$train_control,
    tuneGrid = config$tune_grid,
    num.trees = 500,
    importance = 'impurity',
    metric = "Kappa"
  ))
  stopCluster(cl); registerDoSEQ()

  match_idx <- match(key_new, pi2_map$key)
  pred_vals <- predict(pi2_model, key_new_df)

  result <- character(nrow(x1_new))
  matched <- !is.na(match_idx)
  result[matched] <- pi2_map$val[match_idx[matched]]
  result[!matched] <- level[pred_vals[!matched]]

  return(result)
}

# function for obtaining mapping pi1_opt
pi1_fnc = function(
  pi1_map,
  x1_new, e_new){
  
  key_new_df = data.frame(x1_new, e = e_new)
  key_new = do.call(paste, c(key_new_df, sep = "_"))

  key_df = pi1_map$key_df
  val_df = data.frame(a = as.factor(pi1_map$val))

  level = levels(val_df$a)
  if(length(level) == 1) {
    return(rep(level[1], nrow(x1_new)))
  } else {
    design = cbind(key_df, val_df)

    num_cores <- detectCores() - 1 
    cl <- makeCluster(num_cores); registerDoParallel(cl)
    config <- set_getpi(key_df)
    pi1_model <- train(
      a ~ .,
      data = design,
      method = "ranger",
      trControl = config$train_control,
      tuneGrid = config$tune_grid,
      num.trees = 500,
      importance = 'impurity',
      metric = "Kappa"
    )
    stopCluster(cl); registerDoSEQ()

    match_idx <- match(key_new, pi1_map$key)
    pred_vals <- predict(pi1_model, key_new_df)

    result <- character(nrow(x1_new))
    matched <- !is.na(match_idx)
    result[matched] <- pi1_map$val[match_idx[matched]]
    result[!matched] <- level[pred_vals[!matched]]

    return(result)
  }
}


#evaluate policy value (input seed should match training seed)
evaluateValue = function(
  pi1_map, pi2_map, n=100, seed=42, lab='op', run='seed=42_n=600',
  alpha01, alpha1, alpha02, alpha2, 
  beta01, beta1, beta02, beta2, 
  lambda1, lambda2, gamma01, gamma11, gamma02, gamma12) {

  #generating neccesary storage objects
  seed_new = seed + 1000
  set.seed(seed_new)
  w1_new = w2_new = matrix(nrow=n, ncol=12)
  x2_new = y_new = w1R_new = w2R_new = matrix(nrow=n, ncol=3)
  b1_new = b2_new = rep(0, n)
  tau_dists = matrix(nrow=n, ncol=nrow(allPerms))
  b1probs = b2probs = matrix(nrow=n, ncol=7)
  
  Xdist_n = 10

  #simulating first time point
  if(lab == 'op') {
    v_new = matrix(rnorm(2*n), ncol=2)
    v_extend_new = cbind(rep(1,n), (v_new))
    e_new = t(apply(v_new, 1, softmax))
  } 
  if(lab == 'mis') {
    v_extend_new = e_new = simulate_dirichlet(n, c(1,1,1))
  }
  eR_new = t(apply(e_new, 1, order))
  for (j in 1:ncol(w1_new)) w1_new[,j] = rbinom(n, 1, sigmoid(v_extend_new %*% c(beta01[j], beta1[,j])))
  x1_new =  matrix(rbinom(3*n, Xdist_n, 0.5), ncol=3) 
  for (j in 1:ncol(tau_dists)) tau_dists[,j] = apply(eR_new, 1, tau_metric, pi=allPerms[j,])
  w1R_probs = exp(-lambda1*tau_dists) / rowSums(exp(-lambda1*tau_dists))
  for (i in 1:nrow(w1R_new)) w1R_new[i,] = allPerms[sample(x = seq(1,6), size=1, prob = w1R_probs[i,]),]
  w1R_sparse_new = factor(rowSums(t(t(w1R_new) * c(20,5,1))), labels=seq(1,6))
  
  colnames(w1_new) = paste('W1', 1:ncol(w1), sep = '')
  w1R_sparse_new = data.frame(W1R = w1R_sparse_new)
  a1_opt_label = pi1_fnc(pi1_map = pi1_map, x1_new = x1_new, e_new = v_extend_new)

  a1_opt_sparse <- as.data.frame(matrix(0, nrow = length(a1_opt_label), ncol = 4))
  colnames(a1_opt_sparse) <- paste0("a", 1:4)
  a1_opt_sparse[cbind(seq_along(a1_opt_label), match(a1_opt_label, colnames(a1_opt_sparse)))] <- 1

  #simulating second time point
  for (j in 1:ncol(w2_new)) w2_new[,j] = rbinom(n, 1, sigmoid(v_extend_new %*% c(beta02[j], beta2[,j])))
  w2R_probs = exp(-lambda2*tau_dists) / rowSums(exp(-lambda2*tau_dists))
  for (i in 1:nrow(w2R_new)) w2R_new[i,] = allPerms[sample(x = seq(1,6), size=1, prob = w2R_probs[i,]),]
  w2R_sparse_new = factor(rowSums(t(t(w2R_new) * c(20,5,1))), labels=seq(1,6))
  shift_x1_new = apply(x1_new, 2, mean); scale_x1_new = apply(x1_new, 2, sd)
  for (j in 1:ncol(x2_new)) x2_new[,j] = rbinom(n, Xdist_n, sigmoid(gamma01[,j]%*%t(a1_opt_sparse)+((x1_new[,j]-shift_x1_new[j])/scale_x1_new[j])*gamma11[,j]%*%t(a1_opt_sparse))) 
  for (k in 2:(ncol(b1probs)-1)) b1probs[,k] = sigmoid(alpha01[k] - alpha1*rowSums(x2_new*e_new))-sigmoid(alpha01[k-1] - alpha1*rowSums(x2_new*e_new))
  b1probs[,1] = sigmoid(alpha01[1] - alpha1*rowSums(x2_new*e_new)); b1probs[,7] = 1-rowSums(b1probs[,1:6])
  b1_new = glmnet::rmult(b1probs)
  
  colnames(w2_new) = paste('W2', 1:ncol(w2_new), sep = '')
  w2R_sparse_new = data.frame(W2R = w2R_sparse_new)
  a2_opt_label = pi2_fnc(pi2_map = pi2_map, x1_new = x1_new, x2_new = x2_new, a1_label = data.frame(a1_label = a1_opt_label), e_new = v_extend_new)
  a2_opt_sparse <- as.data.frame(1L * sapply(c("a1", "a2", "a3", "a4"), grepl, a2_opt_label))

  #simulating utilities
  shift_x2_new = apply(x2_new, 2, mean); scale_x2_new = apply(x2_new, 2, sd)
  for (j in 1:ncol(y_new)) y_new[,j] = rbinom(n, Xdist_n, sigmoid(gamma02[,j]%*%t(a2_opt_sparse)+((x2_new[,j]-shift_x2_new[j])/scale_x2_new[j])*gamma12[,j]%*%t(a2_opt_sparse))) 
  for (k in 2:(ncol(b2probs)-1)) b2probs[,k] = sigmoid(alpha02[k] - alpha2*rowSums(y_new*e_new))-sigmoid(alpha02[k-1] - alpha2*rowSums(y_new*e_new))
  b2probs[,1] = sigmoid(alpha02[1] - alpha2*rowSums(y_new*e_new)); b2probs[,7] = 1-rowSums(b2probs[,1:6])
  b2_new = glmnet::rmult(b2probs)
  
  #fitted value
  return(c(mean(rowSums(y_new*e_new)), mean(b2_new), mean(y_new)))
}

