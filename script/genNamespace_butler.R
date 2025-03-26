

# ----------------------------------------
#        Algorithm Helper Functions
# ----------------------------------------

exp_e = function(theta, w1_project, vSim, eSim) {
  beta0 = theta[1:p]; beta1 = theta[(p+1):(2*p)]
  w1_probs = exp(colSums(dbinom(w1_project, 1, sigmoid(beta0 + beta1%*%t(vSim)), log=T)))
  num = cbind(colMeans(eSim[,1] * w1_probs), colMeans(eSim[,2] * w1_probs))
  denom = colMeans(w1_probs)
  cond_e = num/denom
  return(cond_e)
}

cond_y = function(y, x1, a1) {
    mean_y <- array(0, dim = c(nrow(y), ncol(y)))
    y_models <- list()
    for(j in 1:ncol(y)) {
        y_models[[j]] <- lm(y[,j] ~ 1 + x1 + a1 + x1:a1)
        mean_y[,j] <- predict(y_models[[j]])
    }

    return(list(mean_y = mean_y, y_models = y_models))
}

# estimate DTR
piK = function(x1, w1, a1, y, thetaEst, w1_project, vSim, eSim) {
  # get cond_y model
  res = cond_y(y, x1, a1)
  mean_y = res$mean_y; y_models = res$y_models

  # get cond_e
  mean_e = exp_e(thetaEst, w1_project, vSim, eSim)

  # based on utility, get pi_opt
  A1_set = unique(a1)
  utilities = matrix(nrow=length(x1), ncol=length(A1_set))
  for (j in 1:length(A1_set)) {
    a1_broad = rep(A1_set[j], nrow(x1))
    mean_y_pred = matrix(nrow = nrow(x1), ncol = 2)
    for(d in 1:ncol(y)) {
      mean_y_pred[,d] = predict(y_models[[d]], newdata = data.frame(x1 = x1, a1 = a1_broad))
    }
    utilities[,j] = rowSums(mean_y_pred * mean_e)
  }
  q1_max = apply(utilities, 1, max)
  a1_max = A1_set[max.col(utilities, ties.method = "first")]

  # return pi_opt mapping
  pi1_map = list(val = paste0('a', a1_max), key_df = data.frame(x1 = x1, w1 = w1))

  return(list(q1_max=q1_max, a1_max = a1_max, pi1_map = pi1_map))
}

piK_naive = function(x1, w1, a1, y) {
  # get cond_y model
  res = cond_y(y, x1, a1)
  mean_y = res$mean_y; y_models = res$y_models

  # get cond_e
  mean_e = array(1/dim(y)[2], dim=c(nrow(x1), dim(y)[2]))

  # based on utility, get pi_opt
  A1_set = unique(a1)
  utilities = matrix(nrow=length(x1), ncol=length(A1_set))
  for (j in 1:length(A1_set)) {
    a1_broad = rep(A1_set[j], nrow(x1))
    mean_y_pred = matrix(nrow = nrow(x1), ncol = 2)
    for(d in 1:ncol(y)) {
      mean_y_pred[,d] = predict(y_models[[d]], newdata = data.frame(x1 = x1, a1 = a1_broad))
    }
    utilities[,j] = rowSums(mean_y_pred * mean_e)
  }
  q1_max = apply(utilities, 1, max)
  a1_max = A1_set[max.col(utilities, ties.method = "first")]

  # return pi_opt mapping
  pi1_map = list(val = paste0('a', a1_max), key_df = data.frame(x1 = x1, w1 = w1))

  return(list(q1_max=q1_max, a1_max = a1_max, pi1_map = pi1_map))
}

piK_sat = function(x1, w1, a1, b1, thetaEst, w1_project, vSim, eSim) {
  # get cond_y model
  res = cond_y(as.matrix(b1), x1, a1)
  mean_y = res$mean_y; y_models = res$y_models

  # get cond_e
  mean_e = exp_e(thetaEst, w1_project, vSim, eSim)

  # based on utility, get pi_opt
  A1_set = unique(a1)
  utilities = matrix(nrow=length(x1), ncol=length(A1_set))
  for (j in 1:length(A1_set)) {
    a1_broad = rep(A1_set[j], nrow(x1))
    mean_y_pred = matrix(nrow = nrow(x1), ncol = ncol(as.matrix(b1)))
    for(d in 1:ncol(as.matrix(b1))) {
      mean_y_pred[,d] = predict(y_models[[d]], newdata = data.frame(x1 = x1, a1 = a1_broad))
    }
    utilities[,j] = rowSums(mean_y_pred)
  }
  q1_max = apply(utilities, 1, max)
  a1_max = A1_set[max.col(utilities, ties.method = "first")]

  # return pi_opt mapping
  pi1_map = list(val = paste0('a', a1_max), key_df = data.frame(x1 = x1, w1 = w1))

  return(list(q1_max=q1_max, a1_max = a1_max, pi1_map = pi1_map))
}

piK_known = function(x1, w1, a1, y, v) {
  # get cond_y model
  res = cond_y(y, x1, a1)
  mean_y = res$mean_y; y_models = res$y_models

  # get cond_e
  mean_e = cbind(pnorm(v), 1-pnorm(v))

  # based on utility, get pi_opt
  A1_set = unique(a1)
  utilities = matrix(nrow=length(x1), ncol=length(A1_set))
  for (j in 1:length(A1_set)) {
    a1_broad = rep(A1_set[j], nrow(x1))
    mean_y_pred = matrix(nrow = nrow(x1), ncol = 2)
    for(d in 1:ncol(y)) {
      mean_y_pred[,d] = predict(y_models[[d]], newdata = data.frame(x1 = x1, a1 = a1_broad))
    }
    utilities[,j] = rowSums(mean_y_pred * mean_e)
  }
  q1_max = apply(utilities, 1, max)
  a1_max = A1_set[max.col(utilities, ties.method = "first")]

  # return pi_opt mapping
  pi1_map = list(val = paste0('a', a1_max), key_df = data.frame(x1 = x1, v = v))

  return(list(q1_max=q1_max, a1_max = a1_max, pi1_map = pi1_map))
}

# fit estiamted DTR
piK_fnc = function(pi1_map, x1_new, w1_new) {

  key_new_df = data.frame(x1 = x1_new, w1 = w1_new)
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
      metric = "Kappa",
      replace = FALSE,                     
      sample.fraction = 1
    )
    stopCluster(cl); registerDoSEQ()

    result = predict(pi1_model, newdata = key_new_df)
    return(result)
  }
}

piK_known_fnc = function(pi1_map, x1_new, v_new) {

  key_new_df = data.frame(x1 = x1_new, v = v_new)
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
      metric = "Kappa",
      replace = FALSE,                     
      sample.fraction = 1
    )
    stopCluster(cl); registerDoSEQ()

    result = predict(pi1_model, newdata = key_new_df)
    return(result)
  }
}

# evaluate DTR (simulated/estimated V)
evaluateValue = function(
    pi1_map, n, p, seed, run, 
    alpha0, alpha1, beta0, beta1, delta0, delta1, sigma
) {
    
    seed = seed + 1000
    set.seed(seed)

    v_new = rnorm(n)
    w1_new = matrix(nrow = n, ncol = p)
    for (j in 1:p) w1_new[,j] = rbinom(n, 1, sigmoid(beta0[j] + beta1[j]*v_new))
    x1_new = matrix(nrow = n, ncol = 5)
    for (j in 1:5) x1_new[,j] = rnorm(n)

    a1_opt_label = piK_fnc(pi1_map, x1_new, w1_new)
    a1_opt = as.numeric(gsub("a", "", a1_opt_label))

    y_new = matrix(nrow=n, ncol=2)
    for (j in 1:2) y_new[,j] = rnorm(n, cbind(1,x1_new)%*%delta0[,j] + a1_opt*cbind(1,x1_new)%*%delta1[,j], sigma)
    e_new = cbind(pnorm(v_new), 1-pnorm(v_new))
    u_new = rowSums(e_new*y_new)
    u_shift = range(u_new)
    u_tf_new = 6 * (u_new - u_shift[1])/(u_shift[2]-u_shift[1]) - 3
    b1_new = rpois(n, exp(alpha0 + alpha1*u_tf_new))

    return(c(mean(rowSums(y_new*e_new)), mean(b1_new), mean(y_new)))
}

evaluateValue_known = function(
    pi1_map, n, p, seed, run, 
    alpha0, alpha1, beta0, beta1, delta0, delta1, sigma
) {
    
    seed = seed + 1000
    set.seed(seed)

    v_new = rnorm(n)
    w1_new = matrix(nrow = n, ncol = p)
    for (j in 1:p) w1_new[,j] = rbinom(n, 1, sigmoid(beta0[j] + beta1[j]*v_new))
    x1_new = matrix(nrow = n, ncol = 5)
    for (j in 1:5) x1_new[,j] = rnorm(n)

    a1_opt_label = piK_known_fnc(pi1_map, x1_new, v_new)
    a1_opt = as.numeric(gsub("a", "", a1_opt_label))
    table(a1_opt_label)

    y_new = matrix(nrow=n, ncol=2)
    for (j in 1:2) y_new[,j] = rnorm(n, cbind(1,x1_new)%*%delta0[,j] + a1_opt*cbind(1,x1_new)%*%delta1[,j], sigma)
    e_new = cbind(pnorm(v_new), 1-pnorm(v_new))

    u_new = rowSums(e_new*y_new)
    u_shift = range(u_new)
    u_tf_new = 6 * (u_new - u_shift[1])/(u_shift[2]-u_shift[1]) - 3
    b1_new = rpois(n, exp(alpha0 + alpha1*u_tf_new))

    return(c(mean(rowSums(y_new*e_new)), mean(b1_new), mean(y_new)))
}
