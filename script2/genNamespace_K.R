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
    number = 5,
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
  y, xK, aK, model = "rf",
  num_trees = 500, cv_folds = 5,
  allow_parallel = TRUE, verbose_iter = FALSE
) {
  n <- nrow(y)
  J <- ncol(y)
  mean_y <- matrix(0, n, J)
  y_models <- vector("list", J)

  for (j in seq_len(J)) {
    yj  <- y[, j]; xKj <- xK[, j]
    X <- data.frame(xKj = xKj, aK = aK, check.names = FALSE)

    fit <- fit_rf(
      x_df = X,
      y = yj,
      num_trees = num_trees,
      allow_parallel = allow_parallel,
      verbose_iter = verbose_iter
    )
    y_models[[j]] <- fit
    mean_y[, j] <- as.numeric(predict(fit, newdata = X))
  }
  ref <- data.frame(xKj = xK[, 1], aK = aK, check.names = FALSE)
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
  xK, aK, y_models, type = "rf"
) {
  J <- length(y_models); n <- nrow(xK)
  mean_y <- matrix(0, n, J)

  for (j in seq_len(J)) {
    xKj <- xK[, j]
    X <- data.frame(xKj = xKj, aK = aK, check.names = FALSE)
    mean_y[, j] <- as.numeric(predict(y_models[[j]], newdata = X))
  }
  mean_y
}



# =========================
# Calculate Conditional E 
# =========================
cond_exp <- function(
  K, W, X, B, vSim, eSim,
  alpha0, alpha1, beta0, beta1
) {

  n <- dim(X)[1]; nSim <- nrow(eSim); dE <- ncol(eSim)
  log_prob <- matrix(0, nSim, n)

  # W likelihood
  for (k in 1:K) {
    for (j in 1:dim(W)[2]) {
      pkj <- sigmoid(beta0[j, k] + vSim %*% beta1[j, k, ])
      p_w <- matrix(pkj, nrow = nSim, ncol = n)
      wjk <- matrix(W[, j, k], nrow = nSim, ncol = n, byrow = TRUE)
      log_prob <- log_prob +
        wjk * log(pmax(p_w, .Machine$double.xmin)) +
        (1 - wjk) * log(pmax(1 - p_w, .Machine$double.xmin))
    }
  }

  ## B likelihood
  for (k in 1:(K-1)) {
    rel_odds <- alpha1[k] * (X[,,k] %*% t(eSim))
    rel_odds <- t(rel_odds) 

    bprobs <- array(0, dim = c(nSim, n, 3))
    p1 <- sigmoid(alpha0[1, k] - rel_odds)
    p2 <- sigmoid(alpha0[2, k] - rel_odds)
    bprobs[, , 1] <- p1
    bprobs[, , 2] <- p2 - p1
    bprobs[, , 3] <- 1 - p2
    idx <- cbind(
      rep(1:nSim, times = n),
      rep(1:n, each = nSim),
      as.vector(B[, k]))
    log_prob <- log_prob + log(pmax(bprobs[idx], .Machine$double.xmin))
  }

  # Posterior mean of E
  max_lp <- apply(log_prob, 2, max)
  wts <- exp(sweep(log_prob, 2, max_lp))
  denom <- colSums(wts)

  mean_e <- t(wts) %*% eSim / denom
  colnames(mean_e) <- paste0("E", 1:dE)

  return(mean_e)
}



# =========================
#  DTR Evaluation Function 
# =========================

# convert to one-hot from labels "a1"..."a4"
.one_hot_a <- function(a_label) {
  idx <- as.integer(sub("a", "", a_label))
  out <- matrix(0, nrow = length(idx), ncol = 4)
  out[cbind(seq_along(idx), idx)] <- 1
  return(out)
}


evaluateValue <- function(
  q_model_list,
  n_test, test_seed, K,
  alpha0, alpha1, beta0, beta1, gamma0, gamma1,
  alpha0_opt, alpha1_opt, beta0_opt, beta1_opt,
  vextendSim, eSim,
  type = "ours"
) {
  set.seed(test_seed)
  n <- n_test

  num_w <- 2; num_v <- 2; num_b = 3
  num_y <- num_v + 1; num_alpha = num_b - 1

  A_set <- paste0("a", 1:4)
  W_new <- array(0, dim = c(n, num_w, K))
  X_new <- array(0, dim = c(n, num_y, K + 1))
  A_opt_onehot <- array(0, dim = c(n, 4, K))
  B_new <- matrix(0, n, K)
  bprobs <- array(0, dim = c(n, num_b, K))

  # Generate true latent preferences
  V_new = matrix(rnorm(num_v*n), ncol=num_v)
  E_new = t(apply(V_new, 1, softmax))
  
  # baseline
  X_new[, , 1] <- matrix(rbinom(n * num_y, 10, 0.5), nrow = n)
  A_opt_label <- matrix(NA_character_, nrow = n, ncol = K)

  for (k in 1:K) {
    # (1) generate W_new
    for (j in 1:num_w) W_new[,j,k] <- rbinom(n, 1, sigmoid(beta0[j,k] + V_new %*% beta1[j,k,]))

    # (2) choose action at time k
    # Construct Hk
    if (type == 'known') {
      if(k == 1) {
        Hk_df <- data.frame(Xk = X_new[,,1:k], E = E_new)
      } else {
        Hk_df <- data.frame(
          Xk = X_new[,,1:k], 
          A_hist_label = A_opt_label[,1:(k-1)], E = E_new)
        colnames(Hk_df)
      }
    } else {
      if(k == 1) {
        Hk_df <- data.frame(Xk = X_new[,,1:k], Wk = W_new[,,1:k])
      } else {
        Hk_df <- data.frame(
          Xk = X_new[,,1:k], Wk = W_new[,,1:k], 
          A_hist_label = A_opt_label[,1:(k-1)], Bk = B_new[,1:(k-1)])
        colnames(Hk_df)
      }
    }

    ## if not the last decision time
    if (k < K || (type == 'sat' & k == K)) {
      
      utilities <- matrix(0, nrow = n, ncol = 4)
      for (j in 1:length(A_set)) {
        aK_label_dum = rep(A_set[j], nrow(Hk_df))
        covars_j <- data.frame(Hk_df, Ak = aK_label_dum, check.names = FALSE)
        utilities[, j] <- predict(q_model_list[[k]], newdata = covars_j)
      }
      best_a <- max.col(utilities, ties.method = "first")
      A_opt_label[,k] <- paste0("a",best_a)
    
    # If is the last decision time
    } else {
      y_models <- q_model_list[[K]]

      mean_e <- cond_exp(
              K = k, W = W_new[,,1:k,drop=FALSE], X = X_new[,,1:k,drop=FALSE], 
              B = B_new[,1:(k-1),drop=FALSE],
              vSim = vextendSim[, -1, drop = FALSE], eSim = eSim,
              alpha0 = alpha0_opt, alpha1 = alpha1_opt, 
              beta0 = beta0_opt, beta1 = beta1_opt)

      utilities <- matrix(0, nrow = n, ncol = 4)
      for (j in 1:length(A_set)) {
        aK_label_dum = rep(A_set[j], n)
        mean_y_pred = cond_y_predict(xK = X_new[,,k], aK = aK_label_dum, y_models, type = 'rf')
        utilities[,j] = rowSums(mean_y_pred * mean_e)
      }
      best_a <- max.col(utilities, ties.method = "first")
      A_opt_label[,k] <- paste0("a",best_a)
    }

    # (3) generate X_{k+1}
    xk <- X_new[,,k,drop = TRUE]
    shift_x <- colMeans(xk); scale_x <- apply(xk, 2, sd)
    XS <- sweep(sweep(xk, 2, shift_x, "-"), 2, scale_x, "/")
    A_opt_onehot[,,k] = .one_hot_a(A_opt_label[,k])

    for (j in 1:num_y) {
      probs_j = rowSums(A_opt_onehot[,,k] * t(gamma0[,j,k] + t(XS[,j,drop=F]%*%t(gamma1[,j,k]))))
      X_new[,j,k+1] = rbinom(n, 10, sigmoid(probs_j))
    }

    # (4) generate B_k
    rel_odds = alpha1[k]*rowSums(X_new[,,k+1]*E_new)
    bprobs[,1,k] = sigmoid(alpha0[1,k] - rel_odds)
    for (j in 2:num_alpha) bprobs[,j,k] = sigmoid(alpha0[j,k]-rel_odds) - sigmoid(alpha0[j-1,k]-rel_odds)
    bprobs[,num_b,k] = 1 - rowSums(bprobs[,1:num_alpha,k])
    B_new[,k] = glmnet::rmult(bprobs[,,k])
  }
  y_new <- X_new[,,K + 1]

  return(c(
    mean(rowSums(y_new * E_new)),
    mean(B_new[, K]),
    mean(y_new)
  ))
}

