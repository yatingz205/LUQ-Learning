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
V = np$load(makeTitle('V')); W = np$load(makeTitle('W'))
A = np$load(makeTitle('A')); X = np$load(makeTitle('X'))
y = np$load(makeTitle('y')); B = np$load(makeTitle('B'))
vSim = np$load(makeTitle('vSim'))
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

# get conditional E
cond_exp = function(
  K, W, X, A, B, vSim, alpha0, alpha1, beta0, beta1, est_behavior = FALSE) {

  num_w = 2; num_v = 2; num_b = 3
  num_y = num_v + 1; num_alpha = num_b - 1
  n = dim(X)[1]; nSim = nrow(vSim);
  eSim = t(apply(vSim, 1, softmax))
  log_prob <- matrix(0, nSim, n)

  # for X1
  comps_x1_mat = cbind(dbinom(X[, 1, 1], 10, prob = 0.5), dbinom(X[, 2, 1], 10, 0.5), dbinom(X[, 3, 1], 10, 0.5))
  x1_prod = apply(comps_x1_mat, 1, prod)
  x1_comps <- matrix(rep(x1_prod, each=nSim), nrow=nSim, ncol=n, byrow=TRUE)
  log_prob = log_prob + log(x1_comps)
  
  for (k in 1:K) {
    # W components
    for (j in 1:num_w) {
      wj_comps = matrix(0, nSim, n)
      pkj = sigmoid(beta0[j,k] + vSim %*% beta1[j,k,])
      for(i in 1:n) wj_comps[,i] = (dbinom(W[i,j,k], size=1, prob=pkj, log=FALSE))
      log_prob = log_prob + log(pmax(wj_comps, .Machine$double.xmin))
    }
    # A components
    log_prob <- log_prob + log(matrix(1/4, nSim, n))
    # X components (k > 1)
    if (k > 1) {
      shift_x = apply(X[,,k], 2, mean); scale_x = apply(X[,,k], 2, sd)
      XS <- t((t(X[,,k]) - shift_x) / scale_x)
      for (j in 1:num_y) {
        tmp_mat <- matrix(rep(gamma0[, j, k], n), nrow=n, byrow=TRUE) + outer(XS[, j], gamma1[, j, k])
        probs_j <- rowSums(A[,,k] * tmp_mat)
        p_vec = dbinom(X[, j, k+1], size=10, prob=sigmoid(probs_j), log=TRUE)
        log_prob = log_prob + matrix(rep(p_vec, each = nSim), nrow = nSim, ncol = n, byrow = TRUE)
      }
    }
    # B components
    if (k > 1) {
      bprobs = array(0, dim=c(num_b, nSim, n))
      rel_odds <- alpha1[k-1] * (X[,,k-1] %*% t(eSim))
      bprobs[1, , ] <- t(sigmoid(matrix(alpha0[1, k-1],n,nSim) - rel_odds))
      for (j in 2:num_alpha) {
        bprobs[j,,] <- t(sigmoid(matrix(alpha0[j,k-1],n,nSim)-rel_odds) - sigmoid(matrix(alpha0[j-1,k-1],n,nSim)-rel_odds))
      }
      bprobs[num_b,,] <- matrix(1,nSim,n) - (bprobs[1,,] + bprobs[2,,])

      p_vec <- bprobs[cbind(B[, k-1], rep(1:nSim, each = n), rep(1:n, times = nSim))]
      p_vec_clamped <- pmax(p_vec, .Machine$double.xmin) # clamped to avoid -Inf
      p_mat = matrix(p_vec_clamped, nrow = nSim, ncol = n)
      log_prob <- log_prob + log(p_mat)
    }
  }

  #aggregation (gets pr(Hk|V))
  probs = exp(log_prob)
  mean_e = array(0, dim=c(n,3))
  for (j in 1:3) mean_e[,j] = colSums(eSim[,j] * probs) / colSums(probs)
  return(mean_e)
}

# Fit the estiamted policy
pik_fnc = function(pik_map, Hk_new) {

  val_df = data.frame(a = as.factor(pik_map$val))
  level = levels(val_df$a)
  if(length(level) == 1) {
    return(rep(level[1], nrow(Hk_new)))
  } else {
    design = cbind(pik_map$key_df, val_df)
    num_cores <- detectCores() - 1 
    cl <- makeCluster(num_cores); registerDoParallel(cl)
    config <- set_getpi(pik_map$key_df)
    pik_model <- train(
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
    pred_vals <- predict(pik_model, Hk_new)

    return(pred_vals)
  }
}

# get Value of estimated DTR
evaluateValue = function(
    pi_mapList, n, K, seed, run, 
    alpha0, alpha1, beta0, beta1) {
    
    seed_new = seed + 1000
    set.seed(seed_new)

    #creating neccesary storage objects
    num_w = 2; num_v = 2; num_b = 3
    num_y = num_v + 1; num_alpha = num_b - 1
    W_new = array(dim=c(n, num_w, K))
    X_new = array(dim=c(n, num_y, K+1))
    A_opt = array(dim=c(n, 4, K))
    B_new = array(dim=c(n, K))
    bprobs = array(dim=c(n, num_b, K))
    shift_x = scale_x = matrix(nrow=3, ncol=K)

    #simulating trajectories
    V_new = matrix(rnorm(num_v*n), ncol=num_v)
    E_new = t(apply(V_new, 1, softmax))
    X_new[,,1] = rbinom(num_y*n, 10, 0.5)
    A_sparse = matrix(sample(c(1,2,3,4), K*n, replace=T), ncol=K)
    for (k in 1:K) {
        for (j in 1:num_w) W_new[,j,k] = rbinom(n, 1, sigmoid(beta0[j,k] + V_new %*% beta1[j,k,]))
        if(k>1){
          Hk_new = data.frame(X=X_new[,,1:k], W=W_new[,,1:k], B=B_new[,1:k-1], A=A_opt[,,1:k-1])
          colnames(Hk_new) <- gsub("\\.+", "_", colnames(Hk_new))
        } else {
          Hk_new = data.frame(X=X_new[,,1:k], W=W_new[,,1:k])
          colnames(Hk_new) <- gsub("\\.+", "_", colnames(Hk_new))
        }
        ak_opt_label = pik_fnc(pik_map = pi_mapList[[k]], Hk_new = Hk_new)
        ak_opt_sparse <- matrix(0, nrow = length(ak_opt_label), ncol = 4)
        ak_opt_sparse[cbind(seq_along(ak_opt_label), match(ak_opt_label, paste0("a", 1:4)))] <- 1
        A_opt[,,k] = ak_opt_sparse
        shift_x[,k] = apply(X_new[,,k], 2, mean); scale_x[,k] = apply(X_new[,,k], 2, sd)
        XS = t((t(X_new[,,k]) - shift_x[,k])/scale_x[,k])
        for (j in 1:num_y) {
            probs_j = rowSums(A_opt[,,k] * t(gamma0[,j,k] + t(XS[,j,drop=FALSE]%*%t(gamma1[,j,k]))))
            X_new[,j,k+1] = rbinom(n, 10, sigmoid(probs_j))
        }
        rel_odds = alpha1[k]*rowSums(X_new[,,k+1]*E_new)
        bprobs[,1,k] = sigmoid(alpha0[1,k] - rel_odds)
        for (j in 2:num_alpha) bprobs[,j,k] = sigmoid(alpha0[j,k]-rel_odds) - sigmoid(alpha0[j-1,k]-rel_odds)
        bprobs[,num_b,k] = 1 - rowSums(bprobs[,1:num_alpha,k])
        B_new[,k] = glmnet::rmult(bprobs[,,k])
    }
    y_new = X_new[,,K+1]

    return(c(mean(rowSums(y_new*E_new)), mean(B_new[,K]), mean(y_new)))
}
