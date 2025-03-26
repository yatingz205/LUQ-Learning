#helper functions and libraries
suppressMessages({
  library(numDeriv)
  library(doParallel)
  library(caret)
})

sigmoid = function(x) 1/(1+exp(-x))
options(max.print=100, scipen = 8)

#generating model parameters and trajectories
args = commandArgs(trailingOnly=TRUE)
seed = ifelse(length(args) > 0, as.numeric(args[1]), 42)
n = ifelse(length(args) > 1, as.numeric(args[2]), 200)

run = paste0('seed=', seed, '_n=', n)
print(paste('run:', run))
set.seed(seed)

# Generate parameters 
p=10
beta0 = rep(0, p)
beta1 = rnorm(p)
delta0 = matrix(nrow=6, ncol=2)
delta1 = matrix(nrow=6, ncol=2)
delta0[,1] = c(2.5, .2, .25, -.7, -2.5, 2.4)
delta1[,1] = c(1.7, -2.3, 4.5, 6, -7.3, -1.6)
delta0[,2] = 3 - 2*delta0[,1]
delta1[,2] = 3 - 2*delta1[,1]
sigma = 1
alpha0 = 0
alpha1 = 1


# Generate training data
v = rnorm(n)
w1 = matrix(nrow = n, ncol = p)
for (j in 1:p) w1[,j] = rbinom(n, 1, sigmoid(beta0[j] + beta1[j]*v))
x1 = matrix(nrow = n, ncol = 5)
for (j in 1:5) x1[,j] = rnorm(n)
a1 = rbinom(n, 1, 0.5)
y = matrix(nrow=n, ncol=2)
for (j in 1:2) y[,j] = rnorm(n, cbind(1,x1)%*%delta0[,j] + a1*cbind(1,x1)%*%delta1[,j], sigma)
e = cbind(pnorm(v), 1-pnorm(v))
u = rowSums(e*y)
u_shift = range(u)
u_tf = 6 * (u - u_shift[1])/(u_shift[2]-u_shift[1]) - 3
b1 = rpois(n, exp(alpha0 + alpha1*u))
theta = c(beta0, beta1)
# save generated data
save_path <- paste0("simData/butler_data_", run, ".RData")
save(v, w1, x1, a1, y, e, u, b1, file = save_path)
save_path <- paste0("simData/butler_param_", run, ".RData")
save(theta, alpha0, alpha1, delta0, delta1, file = save_path)


# Simulate Data for estimating E[E|H1]
set.seed(seed)
nsamp=1000
vSim = rnorm(nsamp)
eSim = cbind(pnorm(vSim), 1-pnorm(vSim))
w1_project = aperm(array(w1, dim=c(n,p,nsamp)), c(2,3,1))


# Estimating theta
file.path = paste0("estData/butler_param_", run, ".RData")
if(file.exists(file.path)) {
  load(file.path)
} else {
  observedLP <- function(theta) {
    beta0 <- theta[1:p]; beta1 <- theta[(p+1):(2*p)]
    linpred   <- outer(beta1, vSim)
    w1_probs  <- sigmoid(sweep(linpred, 1, beta0, FUN = "+"))
    ll_i <- numeric(n)
    for (i in seq_len(n)) {
      log_lik_mat <- w1[i, ] * log(w1_probs) + (1 - w1[i, ]) * log(1 - w1_probs)
      sum_log <- colSums(log_lik_mat); val_s   <- exp(sum_log)        
      ll_i[i] <- mean(val_s)          
    }
    logLik <- mean(log(ll_i))
    return(10 * logLik)
  }

  observedGP = function(theta) grad(observedLP, theta)

  success <- FALSE; max_attempts <- 5; attempt <- 1
  while(!success && attempt <= max_attempts) {
    cat('Starting attempt numer', attempt, '\n')
    initVals = rnorm(length(theta), sd=0.1)
    result <- tryCatch({
      system.time({
        thetaEst <- optim(
          par = initVals, fn = observedLP, gr = observedGP,
          method = "BFGS", control = list(fnscale = -1, trace = 1, maxit = 200)
        )$par
      })
    }, error = function(e) NULL)

    if (!is.null(result)) {
      success <- TRUE
      elapsed_time <- result
      saveRDS(elapsed_time, paste0("estData/butler_time_", run, ".rds"))
    } else {
      attempt = attempt + 1
    }
  }

  save_path <- paste0("estData/butler_param_", run, ".RData")
  save(thetaEst, file = save_path)
}



# -------------------------------- 
#        Main Script
# --------------------------------

source('genNamespace_butler.R')
source('setGrid.R')

# observational value
initResults = c(mean(rowSums(y*e)), mean(b1), mean(y))

# parameter estimation result
errors = data.frame(
  l1 = sum(abs(as.vector(theta) - as.vector(thetaEst))), 
  mae = mean(abs(as.vector(theta) - as.vector(thetaEst))),
  linf = max(abs(as.vector(theta) - as.vector(thetaEst))))
saveRDS(errors, paste0("estData/butler_errors_", run, ".rds"))


# preference estimation result
e_diff = mean(abs(
  exp_e(theta, w1_project, vSim, eSim) - 
  exp_e(thetaEst, w1_project, vSim, eSim)))
write.csv(e_diff, paste('estData/butler_Eerr_', run, '.csv', sep=''))


# ------------ Butler's Approach -------------

res = piK(x1, w1, a1, y, thetaEst, w1_project, vSim, eSim)
pi1_map = res$pi1_map

newResults = evaluateValue(
  pi1_map, n, p, seed, run, 
  alpha0, alpha1, beta0, beta1, delta0, delta1, sigma)

# policy evaluation result
mat = rbind(initResults, newResults)
colnames(mat) = c('E[U]', 'E[B1]', 'E[Y]')
rownames(mat) = c('observed data', 'estimated DTR'); print(mat)
write.csv(mat, paste('evaData/butlerButler_Mat_', run, '.csv', sep=''))


