#packages and setup
suppressMessages({
  library(doParallel)
  library(glmnet)
  library(ranger)
  library(caret)
})


# ===================================
# Fit Conditional Mean Model with RF
# ===================================
fit_rf <- function(
  x_df, y,
  num_trees = 500,
  cv_folds = 5,
  allow_parallel = FALSE,
  verbose_iter = FALSE,
  metric = "RMSE",
  mtry_seq_step = 3,
  mtry_max_extra = 6,
  min_node_size = c(5, 15, 25),
  splitrule = "variance",
  importance = "impurity"
) {
  stopifnot(is.data.frame(x_df))
  stopifnot(nrow(x_df) == length(y))

  x_df <- as.data.frame(x_df)
  p <- ncol(x_df)
  m0 <- max(1, floor(sqrt(p)))
  m_max <- min(m0 + mtry_max_extra, p)

  tune_grid <- expand.grid(
    mtry = seq(m0, m_max, by = mtry_seq_step),
    min.node.size = min_node_size,
    splitrule = splitrule
  )

  train_control <- caret::trainControl(
    method = "cv",
    number = cv_folds,
    allowParallel = allow_parallel,
    verboseIter = verbose_iter
  )

  caret::train(
    x = x_df,
    y = as.numeric(y),
    method = "ranger",
    num.trees = num_trees,
    importance = importance,
    trControl = train_control,
    tuneGrid = tune_grid,
    metric = metric
  )
}

# =========================
# Fit Conditional Y Model
# =========================
cond_y <- function(
  y, x2, a2, model = "rf",
  num_trees = 500, cv_folds = 5,
  allow_parallel = TRUE, verbose_iter = FALSE
) {
  n <- nrow(y)
  J <- ncol(y)
  mean_y <- matrix(0, n, J)
  y_models <- vector("list", J)

  for (j in seq_len(J)) {
    yj  <- y[, j]; x2j <- x2[, j]
    X <- data.frame(x2j = x2j, a2 = a2, check.names = FALSE)

    fit <- fit_rf(
      x_df = X,
      y = yj,
      num_trees = num_trees,
      cv_folds = cv_folds,
      allow_parallel = allow_parallel,
      verbose_iter = verbose_iter
    )
    y_models[[j]] <- fit
    mean_y[, j] <- as.numeric(predict(fit, newdata = X))
  }

  ref <- data.frame(x2j = x2[, 1], a2 = a2, check.names = FALSE)
  design_names <- colnames(ref)

  list(
    mean_y = mean_y,
    y_models = y_models,
    design_names = design_names,
    type = model
  )
}

# =========================================
# Predict Using Fitted Conditional Y Model
# =========================================
cond_y_predict <- function(
  x2, a2, y_models, type = "rf"
) {
  J <- length(y_models)
  n <- nrow(x2)
  mean_y <- matrix(0, n, J)

  for (j in seq_len(J)) {
    x2j <- x2[, j]
    X <- data.frame(x2j = x2j, a2 = a2, check.names = FALSE)
    mean_y[, j] <- as.numeric(predict(y_models[[j]], newdata = X))
  }
  mean_y
}



# =========================
# Calculate Conditional E 
# =========================
cond_exp = function(
  x2, b1, w1, w2, w1R, w2R, vextendSim, eSim,
  alpha01, alpha1, alpha02, alpha2, 
  beta01, beta1, beta02, beta2, 
  lambda1, lambda2, allPerms
) {

  n = nrow(x2); nSim = nrow(vextendSim)
  
  #vSim components
  eRSim = t(apply(eSim, 1, order))
  tau_dists_sim = array(dim=c(nrow(allPerms), nSim))
  for (j in 1:nrow(allPerms)) {
      tau_dists_sim[j,] = (((eRSim[,2]-eRSim[,1])*(allPerms[j,][2]-allPerms[j,][1]))<0) +
                          (((eRSim[,3]-eRSim[,1])*(allPerms[j,][3]-allPerms[j,][1]))<0) +
                          (((eRSim[,3]-eRSim[,2])*(allPerms[j,][3]-allPerms[j,][2]))<0)
  }
  
  #b1 components
  b1probs = array(dim=c(7,nSim,n))
  u1Sim_broadcast = aperm(array(eSim %*% t(x2), dim=c(nSim, n, 6)), perm=c(3,1,2))
  alpha01_broadcast = array(alpha01, dim = c(6, nSim, n))
  cumProbs1 = sigmoid(alpha01_broadcast-alpha1*u1Sim_broadcast)
  b1probs[2:6,,] = cumProbs1[2:6,,] - cumProbs1[1:5,,] 
  b1probs[1,,] = cumProbs1[1,,]; b1probs[7,,] = 1-cumProbs1[6,,]
  b1_broadcast = aperm(array(model.matrix(~-1+as.factor(b1)), dim=c(n, 7, nSim)), perm=c(2,3,1))
  comps_b1 = colSums(b1probs*b1_broadcast)
  
  #w1 components
  w1_broadcast = aperm(array(w1,dim=c(n,2,nSim)),perm=c(2,3,1))
  w1_probs = sigmoid(vextendSim %*% rbind(beta01, beta1))
  inner_comps_w1 = w1_broadcast*array(log(w1_probs),dim=c(2,nSim,n)) + 
                  (1-w1_broadcast)*array(log(1-w1_probs),dim=c(2,nSim,n))
  comps_w1 = exp(colSums(inner_comps_w1))
  
  #w2 components
  w2_broadcast = aperm(array(w2,dim=c(n,2,nSim)),perm=c(2,3,1))
  w2_probs = sigmoid(vextendSim %*% rbind(beta02, beta2))
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
  probs = comps_w1 * comps_w1R * comps_b1 * comps_w2 * comps_w2R

  mean_e = array(0, dim=c(n,3))
  for (j in 1:3) mean_e[,j] = colSums(eSim[,j]*probs) / colSums(probs)

  return(mean_e)
}



# =========================
#  DTR Evaluation Function 
# =========================

evaluateValue = function(
  # Training data
  x1_train, x2_train, a1_train, a2_train, b1_train, b2_train,
  w1_train, w2_train, w1R_train, w2R_train, y_train,
  # Test data parameters
  n_test, test_seed, lab,
  # True model parameters
  alpha01, alpha1, alpha02, alpha2,
  beta01, beta1, beta02, beta2,
  lambda1, lambda2, gamma01, gamma11, gamma02, gamma12,
  # Estimated parameters
  alpha01_opt, alpha1_opt, alpha02_opt, alpha2_opt,
  beta01_opt, beta1_opt, beta02_opt, beta2_opt,
  lambda1_opt, lambda2_opt,
  # Additional needed parameters
  vextendSim, eSim, allPerms, e_train,
  type = 'known'
) {

  
  # ========== TRAINING PHASE ==========
  a1_label <- paste0('a', factor(apply(a1_train, 1, which.max), levels=c(1,2,3,4)))
  colnames(a2_train) <- paste0('a', 1:ncol(a2_train))
  a2_label  <- apply(a2_train, 1, function(row) {paste0(colnames(a2_train)[which(row == 1)], collapse = "")})
  a1_sparse <- factor(apply(a1_train, 1, which.max))
  A1_set <- unique(a1_label)
  A2_set <- unique(a2_label)

  colnames(w1_train) = paste0('W1', 1:ncol(w1_train))
  colnames(w2_train) = paste0('W2', 1:ncol(w2_train))
  w1R_sparse_train = factor(rowSums(t(t(w1R_train) * c(20,5,1))), labels=seq(1,6))
  w2R_sparse_train = factor(rowSums(t(t(w2R_train) * c(20,5,1))), labels=seq(1,6))
  
  # Fit Y models
  res = cond_y(y_train, x2_train, a2_label, model = 'rf')
  y_models = res$y_models
  
  
  # Compute Q2 values on training data
  if (type == 'naive') {
    mean_e_train = array(1/dim(y_train)[2], dim=c(nrow(x2_train), dim(y_train)[2]))
  } else if (type == 'known') {
    mean_e_train = e_train
  } else if (type == 'ours') {
    mean_e_train = cond_exp(
      x2_train, b1_train, w1_train, w2_train, 
      w1R_train, w2R_train, vextendSim, eSim,
      alpha01_opt, alpha1_opt, alpha02_opt, alpha2_opt,
      beta01_opt, beta1_opt, beta02_opt, beta2_opt,
      lambda1_opt, lambda2_opt, allPerms)
  } else if (type == 'sat') {
    mean_e_train = rep(0, nrow(x2_train))
    covars_b2 <- data.frame(
      x1=x1_train, x2=x2_train, 
      a1_label=a1_label, b1=b1_train, 
      w1=w1_train, w2=w2_train, 
      W1R=w1R_sparse_train, W2R=w2R_sparse_train, a2_label=a2_label)
    b2_model = fit_rf(x_df = covars_b2, y = b2_train)
  } else {
    stop('unsupported type')
  }
  
  utilities_train = matrix(nrow=nrow(x2_train), ncol=length(A2_set))
  for (j in 1:length(A2_set)) {
    a2_label_dum = rep(A2_set[j], nrow(x2_train))

    if (type == 'sat') {
      covars_j = data.frame(
        x1=x1_train, x2=x2_train, 
        a1_label=a1_label, b1=b1_train, 
        w1=w1_train, w2=w2_train, 
        W1R=w1R_sparse_train, W2R=w2R_sparse_train, a2_label=a2_label_dum)
      utilities_train[,j] = predict(b2_model, covars_j)
    } else {
      mean_y_pred = cond_y_predict(x2_train, a2_label_dum, y_models, type = 'rf')
      utilities_train[,j] = rowSums(mean_y_pred * mean_e_train)
    }
  }
  q2_max_train = apply(utilities_train, 1, max)

  # Fit Q1 model
  if (type == 'known') {
    covars_q1 = data.frame(x1=x1_train, e=e_train, a1=a1_label)
    q1_model <- fit_rf(x_df = covars_q1, y = q2_max_train)
  } else {
    covars_q1 = data.frame(x1=x1_train, w1=w1_train, w1R=w1R_sparse_train, a1=a1_label)
    q1_model <- fit_rf(x_df = covars_q1, y = q2_max_train)
  }

  
  # ========== GENERATE TEST DATA UNDER DTR HAT ==========
  set.seed(test_seed)
  n = n_test 

  w1_new = w2_new = matrix(nrow=n, ncol=12)
  x2_new = y_new = w1R_new = w2R_new = matrix(nrow=n, ncol=3)
  b1_new = b2_new = rep(0, n)
  tau_dists = matrix(nrow=n, ncol=nrow(allPerms))
  b1probs = b2probs = matrix(nrow=n, ncol=7)

  # Generate true latent preferences
  if(lab == 'op') {
    v_new = matrix(rnorm(2*n), ncol=2)
    v_extend_new = cbind(rep(1,n), v_new)
    e_new = t(apply(v_new, 1, softmax))
  } else if(lab == 'mis') {  
    v_extend_new = e_new = simulate_dirichlet(n, c(1,1,1))
  }
  eR_new = t(apply(e_new, 1, order))


  # ========== STAGE 1 ==========
  # Generate Stage 1 covariates
  for (j in 1:ncol(w1_new)) w1_new[,j] = rbinom(n, 1, sigmoid(v_extend_new %*% c(beta01[j], beta1[,j])))
  x1_new = matrix(rbinom(3*n, 10, 0.5), ncol=3) 
  for (j in 1:ncol(tau_dists)) tau_dists[,j] = apply(eR_new, 1, tau_metric, pi=allPerms[j,])
  w1R_probs = exp(-lambda1*tau_dists) / rowSums(exp(-lambda1*tau_dists))
  for (i in 1:nrow(w1R_new)) w1R_new[i,] = allPerms[sample(x = seq(1,6), size=1, prob = w1R_probs[i,]),]
  w1R_sparse_new = factor(rowSums(t(t(w1R_new) * c(20,5,1))), labels=seq(1,6))
  
  # Apply Q1 model to get optimal Stage 1 actions
  colnames(w1_new) = paste0('W1', 1:ncol(w1_new))
  utilities_stage1 = matrix(nrow=nrow(x1_new), ncol=length(A1_set))
  for (j in 1:length(A1_set)) {
    a1_label_dum = rep(A1_set[j], nrow(x1_new))
    if (type == 'known') {
      covars_new = data.frame(x1=x1_new, e=e_new, a1=a1_label_dum)
      colnames(covars_new) = colnames(covars_q1)
    } else {
      covars_new = data.frame(x1=x1_new, w1=w1_new, w1R=w1R_sparse_new, a1=a1_label_dum)
    }
    utilities_stage1[,j] = predict(q1_model, covars_new)
  }
  a1_indices = as.numeric(gsub("a", "", A1_set[apply(utilities_stage1, 1, which.max)]))
  a1_opt_new = diag(4)[a1_indices, ]
  

  # ========== STAGE 2 ==========
  for (j in 1:ncol(w2_new)) w2_new[,j] = rbinom(n, 1, sigmoid(v_extend_new %*% c(beta02[j], beta2[,j])))
  colnames(w2_new) = paste0('W2', 1:ncol(w2_new))
  w2R_probs = exp(-lambda2*tau_dists) / rowSums(exp(-lambda2*tau_dists))
  for (i in 1:nrow(w2R_new)) w2R_new[i,] = allPerms[sample(x = seq(1,6), size=1, prob = w2R_probs[i,]),]
  w2R_sparse_new = factor(rowSums(t(t(w2R_new) * c(20,5,1))), labels=seq(1,6))
  
  shift_x1_new = apply(x1_new, 2, mean); scale_x1_new = apply(x1_new, 2, sd)
  for (j in 1:ncol(x2_new)) {
    x2_new[,j] = rbinom(n, 10, sigmoid(
      gamma01[,j] %*% t(a1_opt_new) +
      ((x1_new[,j]-shift_x1_new[j])/scale_x1_new[j]) * gamma11[,j] %*% t(a1_opt_new)
    ))
  }
  
  for (k in 2:(ncol(b1probs)-1)) {
    b1probs[,k] = sigmoid(alpha01[k] - alpha1*rowSums(x2_new*e_new)) - 
                  sigmoid(alpha01[k-1] - alpha1*rowSums(x2_new*e_new))
  }
  b1probs[,1] = sigmoid(alpha01[1] - alpha1*rowSums(x2_new*e_new))
  b1probs[,7] = 1 - rowSums(b1probs[,1:6])
  b1_new = glmnet::rmult(b1probs)
  dim(b1_new) = length(b1_new)
  
  # Compute conditional expectation of preferences 
  mean_e_new = cond_exp( 
    x2_new, b1_new, w1_new, w2_new, 
    w1R_new, w2R_new, 
    vextendSim, eSim,
    alpha01_opt, alpha1_opt, alpha02_opt, alpha2_opt,
    beta01_opt, beta1_opt, beta02_opt, beta2_opt,
    lambda1_opt, lambda2_opt, allPerms
  )
  
  # Compute utilities for all Stage 2 actions
  a1_label_new = paste0('a', a1_indices)
  utilities_stage2 = matrix(nrow=nrow(x2_new), ncol=length(A2_set))
  for (j in 1:length(A2_set)) {
    a2_label_dum = rep(A2_set[j], nrow(x2_new))

    if (type == 'sat') {
      covars_new = data.frame(
        x1=x1_new, x2=x2_new, a1_label=a1_label_new, b1=b1_new, 
        w1=w1_new, w2=w2_new, W1R=w1R_sparse_new, W2R=w2R_sparse_new, 
        a2_label=a2_label_dum)
      utilities_stage2[,j] = predict(b2_model, covars_new)
    } else {
      mean_y_pred = cond_y_predict(x2_new, a2_label_dum, y_models, type = 'rf')
      utilities_stage2[,j] = rowSums(mean_y_pred * mean_e_new)
    }
  }
  A2_mat <- sapply(1:4, function(k) grepl(paste0("a", k), A2_set)) * 1
  colnames(A2_mat) <- paste0("a", 1:4)
  a2_opt_new = A2_mat[max.col(utilities_stage2, ties.method = "first"), , drop=FALSE]
  

  # ========== GENERATE FINAL OUTCOMES ==========
  shift_x2_new = apply(x2_new, 2, mean); scale_x2_new = apply(x2_new, 2, sd)
  for (j in 1:ncol(y_new)) {
    y_new[,j] = rbinom(n, 10, sigmoid(
      gamma02[,j] %*% t(a2_opt_new) + 
      ((x2_new[,j]-shift_x2_new[j])/scale_x2_new[j]) * gamma12[,j] %*% t(a2_opt_new)
    ))
  }
  
  for (k in 2:(ncol(b2probs)-1)) {
    b2probs[,k] = sigmoid(alpha02[k] - alpha2*rowSums(y_new*e_new)) - 
                  sigmoid(alpha02[k-1] - alpha2*rowSums(y_new*e_new))
  }
  b2probs[,1] = sigmoid(alpha02[1] - alpha2*rowSums(y_new*e_new))
  b2probs[,7] = 1 - rowSums(b2probs[,1:6])
  b2_new = glmnet::rmult(b2probs)
  
  # Return performance metrics
  return(c(mean(rowSums(y_new*e_new)), mean(b2_new), mean(y_new)))
}

