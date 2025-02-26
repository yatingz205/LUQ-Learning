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
v = np$load(makeTitle('v')); vSim = np$load(makeTitle('vSim'))
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
alpha01_opt = alpha_opt[1,1:2]; alpha1_opt = alpha_opt[1,3]; alpha02_opt = alpha_opt[2,1:2]; alpha2_opt = alpha_opt[2,3]
lambda1_opt = lambda_opt[1]; lambda2_opt = lambda_opt[2]


# function of fitting the propensity score model
# both of and cond_on should be data frames
cond_pr_a <- function(
  of, cond_on, of_eval, cond_on_eval, model = 'logistic') {
  
  # Validate the 'model' parameter
  if (!(model %in% c('logistic', 'rf'))) {
    stop("Model must be either 'logistic' or 'rf'")
  }

  est_probs <- NULL
  # Iterate over each target variable in 'of'
  for (j in 1:ncol(of)) {
    response <- of[[j]]
    
    # Prepare the modeling data
    data_model <- data.frame(response = factor(response), cond_on)
    n_levels <- length(unique(data_model$response))
    levels <- levels(data_model$response)
    
    if (model == 'logistic') {
      if (n_levels == 2) {
        # Binary Logistic Regression
        if(!all(levels %in% c(0, 1))) stop("Response variable must be coded as 0 and 1")
        fit <- glm(response ~ ., data = data_model, family = binomial)
        probs <- predict(fit, cond_on_eval, type = 'response')
        probs <- ifelse(of_eval[[j]] == 1, probs, 1 - probs)
      } else if (n_levels > 2) {
        # Multinomial Logistic Regression using 'nnet' package
        fit <- nnet::multinom(response ~ ., data = data_model, trace = FALSE)
        probs_matrix <- predict(fit, cond_on_eval, type = 'probs') 
        probs <- diag(probs_matrix[, of_eval[[j]]])
      } else {
        stop("Response variable must have at least two levels")
      }

    } else if (model == 'rf') {
      fit <- randomForest::randomForest(
        response ~ ., data = data_model, 
        ntree = 500, mtry = floor(sqrt(ncol(cond_on))))
      probs_matrix <- predict(fit, newdata = cond_on_eval, type = 'prob') 
      probs <- diag(probs_matrix[, of_eval[[j]]])
    }

    cat("Distribution of est probs :\n")
    print(quantile(probs, probs = c(0, 0.25, 0.5, 0.75, 1), na.rm = TRUE))
    
    # Append the estimated probabilities to the 'est_probs' data frame
    est_probs <- cbind(est_probs, probs)

    j <- j + 1
  }
  return(est_probs)
}

# estimate conditional probability
cond_pr_y = function(y, x2, a2, model = 'rf') {
  pr_y <- array(0, dim = c(nrow(y), ncol(y)))
  y_models <- list()

  if(model == 'betabinom'){
    for (j in 1:ncol(y)) {
      yj <- y[, j]; std_x2j <- (x2[, j] - mean(x2[,j]))/sd(x2[,j])
      response <- cbind(successes = yj, failures = 10 - yj)
      design_mat <- data.frame(a2 = factor(a2), std_x2j = std_x2j)
      
      y_models[[j]] <- glm(response ~ factor(a2) + std_x2j * factor(a2), 
                       family = binomial(link = "logit"), 
                       data = design_mat)
      pr_y[, j] <- predict(y_models[[j]], newdata = design_mat, type = "response")
    }
  } else if(model == 'rf') {
    for(j in 1:ncol(y)) {
      yj <- as.factor(paste0('y', y[, j])); x2j <- x2[, j]
      design_mat <- data.frame(yj = yj, x2j = x2j, a2 = a2)

      cl <- makeCluster(detectCores() - 1); registerDoParallel(cl)
      config <- set_gettransition(design_mat)
      fit <- train(
        as.formula(paste0('yj ~ .')),
        data = design_mat,
        method = "ranger",
        trControl = config$train_control,
        tuneGrid = config$tune_grid,
        num.trees = 500,
        importance = 'impurity',
        metric = "Kappa"
      )
      stopCluster(cl); registerDoSEQ()
      y_models[[j]] <- fit
      preds <- predict(fit, data = design_mat[, -1], type = "prob") 
      pr_y[, j] <- diag(as.matrix(preds[1:nrow(x2), as.numeric(yj)]))
    }
  } 
  return(list(pr_y = pr_y, y_models = y_models))
}

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

#conditional expectation function (using true model form)
# calls cond_pr_y() to get the transition model, and cond_pr_a() to get the propensity model
cond_exp = function(
  x1, x2, a1, a2, b1, b2, w1, w2, w1R, w2R, vSim, 
  alpha01, alpha1, alpha02, alpha2, 
  beta01, beta1, beta02, beta2, 
  lambda1, lambda2, est_behavior = FALSE) {

  n = nrow(x2); nSim = nrow(vSim)

  # x and y components
  comps_x1_mat = cbind(dbinom(x1[, 1], 5, prob = 0.5), dbinom(x1[, 2], 5, 0.5), dbinom(x1[, 3], 5, 0.5))
  comps_x1 = matrix(rep(apply(comps_x1_mat, 1, prod), each = nSim), nrow = nSim, ncol = n, byrow = TRUE)
  shift_x1 = apply(x1, 2, mean); scale_x1 = apply(x1, 2, sd)
  
  #comps_x2_mat = cond_pr_y(x2, x1, a1, model = 'betabinom')$pr_y
  pr_x2 <- matrix(NA_real_, nrow = nrow(x2), ncol = ncol(x2))
  for (j in 1:ncol(x2)) {
    logit_value <- gamma01[, j] %*% t(a1) + ((x1[, j] - shift_x1[j]) / scale_x1[j]) * gamma11[, j] %*% t(a1)
    prob_value <- sigmoid(logit_value); pr_x2[, j] <- dbinom(x2[, j], size = 10, prob = prob_value)
  }
  comps_x2 = matrix(rep(apply(pr_x2, 1, prod), each = nSim), nrow = nSim, ncol = n, byrow = TRUE)

  # a components
  comps_a1_vec = as.array(rep(1/4, n), dim = c(n, 1))
  comps_a1 <- matrix(rep(comps_a1_vec, each = nSim), nrow = nSim, ncol = n, byrow = TRUE)

  if(est_behavior){
    colnames(a2) <- c('a1', 'a2', 'a3', 'a4')
    comps_a2_vec = cond_pr_a(
      of = data.frame(a2 = apply(a2, 1, function(row) paste0(colnames(a2)[which(row == 1)], collapse = ""))),
      cond_on = data.frame(x2 = x2, b1 = b1, a1 = a1),
      of_eval = data.frame(a2 = apply(a2, 1, function(row) paste0(colnames(a2)[which(row == 1)], collapse = ""))),
      cond_on_eval = data.frame(x2 = x2, b1 = b1, a1 = a1),
      model = 'logistic')
  } else {
    p11 = 1/16+1/16*1/2+1/16; p12 = 1/16+1/16*1/6*2+1/16*1/3*2; p2 = 1/16+1/16*1/3+1/16*1/6; p3 = (1/16*1/3+1/16*1/6)*2
    lookup_table <- c(
      "1000" = p11, "0100" = p12, "0010" = p12, "0001" = p12,  
      "1100" = p2, "1010" = p2, "1001" = p2, 
      "0110" = p3, "0101" = p3, "0011" = p3)
    row_keys <- apply(a2, 1, paste0, collapse = "")
    comps_a2_vec = lookup_table[row_keys]
  }
  comps_a2 <- matrix(rep(comps_a2_vec, each = nSim), nrow = nSim, ncol = n, byrow = TRUE)
  
  #vSim components
  eSim = t(apply(vSim, 1, softmax))
  eRSim = t(apply(eSim, 1, order))
  tau_dists_sim = array(dim=c(nrow(allPerms), nSim))
  for (j in 1:nrow(allPerms)) {
      tau_dists_sim[j,] = (((eRSim[,2]-eRSim[,1])*(allPerms[j,][2]-allPerms[j,][1]))<0) +
                          (((eRSim[,3]-eRSim[,1])*(allPerms[j,][3]-allPerms[j,][1]))<0) +
                          (((eRSim[,3]-eRSim[,2])*(allPerms[j,][3]-allPerms[j,][2]))<0)
  }
  
  #b1 components
  b1probs = array(dim=c(3,nSim,n))
  u1Sim_broadcast = aperm(array(eSim %*% t(x2), dim=c(nSim, n, 2)), perm=c(3,1,2))
  alpha01_broadcast = array(alpha01, dim = c(2, nSim, n))
  cumProbs1 = sigmoid(alpha01_broadcast-alpha1*u1Sim_broadcast)
  b1probs[2,,] = cumProbs1[2,,] - cumProbs1[1,,] 
  b1probs[1,,] = cumProbs1[1,,]; b1probs[3,,] = 1-cumProbs1[2,,]
  b1_broadcast = aperm(array(model.matrix(~-1+as.factor(b1)), dim=c(n, 3, nSim)), perm=c(2,3,1))
  comps_b1 = colSums(b1probs*b1_broadcast)
  
  #w1 components
  w1_broadcast = aperm(array(w1,dim=c(n,2,nSim)),perm=c(2,3,1))
  w1_probs = sigmoid(beta01+t(vSim%*%beta1))
  inner_comps_w1 = w1_broadcast*array(log(w1_probs),dim=c(2,nSim,n)) + 
                  (1-w1_broadcast)*array(log(1-w1_probs),dim=c(2,nSim,n))
  comps_w1 = exp(colSums(inner_comps_w1))
  
  #w2 components
  w2_broadcast = aperm(array(w2,dim=c(n,2,nSim)),perm=c(2,3,1))
  w2_probs = sigmoid(beta02+t(vSim%*%beta2))
  inner_comps_w2 = w2_broadcast*array(log(w2_probs),dim=c(2,nSim,n)) + 
                  (1-w2_broadcast)*array(log(1-w2_probs),dim=c(2,nSim,n))
  comps_w2 = exp(colSums(inner_comps_w2))
  
  #pre-processing for rank components
  w1R_broadcast = w2R_broadcast = array(dim=c(nrow(allPerms), n))
  for (j in 1:nrow(allPerms)) {
    w1R_broadcast[j,] = as.numeric(apply(w1R, 1, function(x) sum(x==allPerms[j,])==3))
    w2R_broadcast[j,] = as.numeric(apply(w2R, 1, function(x) sum(x==allPerms[j,])==3))
  }
  
  #w1R and w2R components
  w1R_probs = exp(-lambda1*tau_dists_sim) / rowSums(exp(-lambda1*tau_dists_sim))
  w2R_probs = exp(-lambda2*tau_dists_sim) / rowSums(exp(-lambda2*tau_dists_sim))
  w1R_broadcast = aperm(array(w1R_broadcast, dim=c(nrow(allPerms), n, nSim)), perm=c(1,3,2))
  w2R_broadcast = aperm(array(w2R_broadcast, dim=c(nrow(allPerms), n, nSim)), perm=c(1,3,2))
  comps_w1R = array(rep(w1R_probs, n)[as.logical(w1R_broadcast)], dim=c(nSim,n))
  comps_w2R = array(rep(w2R_probs, n)[as.logical(w2R_broadcast)], dim=c(nSim,n))
  
  #aggregation (gets pr(H2|V))
  probs = comps_x1 * comps_w1 * comps_w1R * comps_a1 * comps_b1 * 
          comps_x2 * comps_a2 * comps_w2 * comps_w2R

  mean_e = array(0, dim=c(n,3))
  for (j in 1:3) mean_e[,j] = colSums(eSim[,j]*probs) / colSums(probs)
  return(mean_e)
}

# function for obtaining mapping pi2_opt
pi2_fnc = function(
  pi2_map,
  x1_new, x2_new, a1_label, b1_new, 
  w1_new, w2_new, w1R_sparse_new, w2R_sparse_new){

  key_new_df = data.frame(
    x1_new, x2_new, a1_label, b1 = b1_new, w1_new, w2_new, w1R_sparse_new, w2R_sparse_new)
  key_new = do.call(paste, c(key_new_df, sep = "_"))

  key_df = pi2_map$key_df
  val_df = data.frame(a = as.factor(pi2_map$val))

  level = levels(val_df$a)
  design = cbind(key_df, val_df)

  num_cores <- detectCores() - 1 
  cl <- makeCluster(num_cores); registerDoParallel(cl)
  config <- set_getpi(key_df)
  pi2_model <- train(
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
  x1_new, w1_new, w1R_sparse_new){
  
  key_new_df = data.frame(x1_new, w1_new, w1R = w1R_sparse_new)
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
#parameters are the true prams used for training to ensure consistency
evaluateValue = function(
  pi1_map, pi2_map, n=100, seed=42, run='seed=42_n=600',
  alpha01, alpha1, alpha02, alpha2, 
  beta01, beta1, beta02, beta2, 
  lambda1, lambda2, gamma01, gamma11, gamma02, gamma12) {

  #generating neccesary storage objects
  seed_new = seed + 1000
  set.seed(seed_new)
  w1_new = w2_new = matrix(nrow=n, ncol=2)
  x2_new = y_new = w1R_new = w2R_new = matrix(nrow=n, ncol=3)
  b1_new = b2_new = rep(0, n)
  tau_dists = matrix(nrow=n, ncol=nrow(allPerms))
  b1probs = b2probs = matrix(nrow=n, ncol=3)
  
  Xdist_n = 10

  #simulating first time point
  v_new = matrix(rnorm(2*n), ncol=2)
  e_new = t(apply(v_new, 1, softmax))
  eR_new = t(apply(e_new, 1, order))
  for (j in 1:ncol(w1_new)) w1_new[,j] = rbinom(n, 1, sigmoid(v_new%*%beta1[,j]))
  x1_new =  matrix(rbinom(3*n, Xdist_n, 0.5), ncol=3) 
  for (j in 1:ncol(tau_dists)) tau_dists[,j] = apply(eR_new, 1, tau_metric, pi=allPerms[j,])
  w1R_probs = exp(-lambda1*tau_dists) / rowSums(exp(-lambda1*tau_dists))
  for (i in 1:nrow(w1R_new)) w1R_new[i,] = allPerms[sample(x = seq(1,6), size=1, prob = w1R_probs[i,]),]
  w1R_sparse_new = factor(rowSums(t(t(w1R_new) * c(20,5,1))), labels=seq(1,6))
  
  colnames(w1_new) = paste('W1', 1:ncol(w1), sep = '')
  w1R_sparse_new = data.frame(W1R = w1R_sparse_new)
  a1_opt_label = pi1_fnc(pi1_map = pi1_map, x1_new = x1_new, w1_new = w1_new, w1R_sparse_new = w1R_sparse_new)

  a1_opt_sparse <- as.data.frame(matrix(0, nrow = length(a1_opt_label), ncol = 4))
  colnames(a1_opt_sparse) <- paste0("a", 1:4)
  a1_opt_sparse[cbind(seq_along(a1_opt_label), match(a1_opt_label, colnames(a1_opt_sparse)))] <- 1

  #simulating second time point
  for (j in 1:ncol(w2_new)) w2_new[,j] = rbinom(n, 1, sigmoid(v_new %*% beta2[,j]))
  w2R_probs = exp(-lambda2*tau_dists) / rowSums(exp(-lambda2*tau_dists))
  for (i in 1:nrow(w2R_new)) w2R_new[i,] = allPerms[sample(x = seq(1,6), size=1, prob = w2R_probs[i,]),]
  w2R_sparse_new = factor(rowSums(t(t(w2R_new) * c(20,5,1))), labels=seq(1,6))
  shift_x1_new = apply(x1_new, 2, mean); scale_x1_new = apply(x1_new, 2, sd)
  for (j in 1:ncol(x2_new)) x2_new[,j] = rbinom(n, Xdist_n, sigmoid(gamma01[,j]%*%t(a1_opt_sparse)+((x1_new[,j]-shift_x1_new[j])/scale_x1_new[j])*gamma11[,j]%*%t(a1_opt_sparse))) 
  for (k in 2:(ncol(b1probs)-1)) b1probs[,k] = sigmoid(alpha01[k] - alpha1*rowSums(x2_new*e_new))-sigmoid(alpha01[k-1] - alpha1*rowSums(x2_new*e_new))
  b1probs[,1] = sigmoid(alpha01[1] - alpha1*rowSums(x2_new*e_new)); b1probs[,3] = 1-rowSums(b1probs[,1:2])
  b1_new = glmnet::rmult(b1probs)
  
  colnames(w2_new) = paste('W2', 1:ncol(w2_new), sep = '')
  w2R_sparse_new = data.frame(W2R = w2R_sparse_new)
  a2_opt_label = pi2_fnc(pi2_map = pi2_map, x1_new = x1_new, x2_new = x2_new, a1_label = data.frame(a1_label = a1_opt_label), 
    b1_new = b1_new, w1_new = w1_new, w2_new = w2_new, w1R_sparse_new = w1R_sparse_new, w2R_sparse_new = w2R_sparse_new) 
  a2_opt_sparse <- as.data.frame(1L * sapply(c("a1", "a2", "a3", "a4"), grepl, a2_opt_label))

  #simulating utilities
  shift_x2_new = apply(x2_new, 2, mean); scale_x2_new = apply(x2_new, 2, sd)
  for (j in 1:ncol(y_new)) y_new[,j] = rbinom(n, Xdist_n, sigmoid(gamma02[,j]%*%t(a2_opt_sparse)+((x2_new[,j]-shift_x2_new[j])/scale_x2_new[j])*gamma12[,j]%*%t(a2_opt_sparse))) 
  for (k in 2:(ncol(b2probs)-1)) b2probs[,k] = sigmoid(alpha02[k] - alpha2*rowSums(y_new*e_new))-sigmoid(alpha02[k-1] - alpha2*rowSums(y_new*e_new))
  b2probs[,1] = sigmoid(alpha02[1] - alpha2*rowSums(y_new*e_new)); b2probs[,3] = 1-rowSums(b2probs[,1:2])
  b2_new = glmnet::rmult(b2probs)
  
  #fitted value
  return(c(mean(rowSums(y_new*e_new)), mean(b2_new), mean(y_new)))
}

