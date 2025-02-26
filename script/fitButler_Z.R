#helper functions and libraries
suppressMessages({
  library(numDeriv)
  library(doParallel)
  library(caret)
})

sigmoid = function(x) 1/(1+exp(-x))
softmax = function(x) c(exp(x),1)/(sum(c(exp(x),1)))
softplus = function(x) log(1+exp(x))
options(max.print=100, scipen = 8)

#generating model parameters and trajectories
args = commandArgs(trailingOnly=TRUE)
seed = ifelse(length(args) > 0, as.numeric(args[1]), 42)
n = ifelse(length(args) > 1, as.numeric(args[2]), 200)

run = paste0('seed=', seed, '_n=', n)
print(paste('run:', run))
set.seed(seed)

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
u = 6 * (u - u_shift[1])/(u_shift[2]-u_shift[1]) - 3
b1 = rpois(n, exp(alpha0 + alpha1*u))
theta = c(beta0, beta1, alpha0, alpha1)
# save generated data
save_path <- paste0("simData/butlerZ_data_", run, ".RData")
save(v, w1, x1, a1, y, e, u, b1, file = save_path)
save_path <- paste0("simData/butlerZ_param_", run, ".RData")
save(theta, delta0, delta1, file = save_path)


#observed LP function
set.seed(seed)
nsamp=1000
vSim = rnorm(nsamp)
eSim = cbind(pnorm(vSim), 1-pnorm(vSim))
b1_project = matrix(rep(b1, nsamp), ncol=n, byrow=T)
w1_project = aperm(array(w1, dim=c(n,p,nsamp)), c(2,3,1))
uSim = (y %*% t(eSim))
uSim = 6 * (uSim - u_shift[1])/(u_shift[2]-u_shift[1]) - 3

observedLP <- function(theta) {
  beta0 <- theta[1:p]; beta1 <- theta[(p+1):(2*p)]
  alpha0 <- theta[(2*p+1)]; alpha1 <- theta[(2*p+2)]
  b1_mean <- exp(alpha0 + alpha1 * uSim) 
  b1_comps_mat <- b1 * log(b1_mean) - b1_mean
  linpred <- outer(beta1, vSim) 
  for(j in seq_len(p)) {linpred[j, ] <- linpred[j, ] + beta0[j]}
  w1_probs <- 1 / (1 + exp(-linpred))
  log_w1_probs <- log(w1_probs); log_one_minus_probs <- log(1 - w1_probs)    
  w1_comps_mat <- matrix(0, n, nsamp)
  for(s in seq_len(nsamp)) {
    logp_s  <- matrix(log_w1_probs[, s], nrow=n, ncol=p, byrow=TRUE)
    log1p_s <- matrix(log_one_minus_probs[, s], nrow=n, ncol=p, byrow=TRUE)
    w1_comps_mat[, s] <- rowSums(w1 * logp_s + (1 - w1) * log1p_s)
  }
  total_mat <- w1_comps_mat + b1_comps_mat
  comps     <- rowMeans(exp(total_mat))
  logLik    <- mean(log(comps))
  penalty <- -exp(-100 * alpha1)
  return(10 * (logLik + penalty))
}


#testing observed LP
initVals = c(rnorm(length(c(beta0,beta1,alpha0)), sd=0.1), rexp(1, sqrt(0.1)))
observedGP = function(theta) grad(observedLP, theta)
elapsed_time <- system.time({thetaEst = optim(
  par = initVals, fn = observedLP, gr = observedGP, 
  method = 'BFGS', control=list(fnscale=-1, trace=1, maxit=200))$par})

save_path <- paste0("estData/butlerZ_param_", run, ".RData")
save(thetaEst, file = save_path)


# ----------------------------------------
#        Algorithm Helper Functions
# ----------------------------------------

source('genNamespace_butler.R')
source('setGrid.R')

# observational value
e = t(sapply(v, softmax))
initResults = c(mean(rowSums(y*e)), mean(b1), mean(y))

# parameter estimation result
errors = data.frame(
  l1 = sum(abs(as.vector(theta) - as.vector(thetaEst))), 
  mae = mean(abs(as.vector(theta) - as.vector(thetaEst))),
  linf = max(abs(as.vector(theta) - as.vector(thetaEst))))
saveRDS(errors, paste0("estData/butlerZ_errors_", run, ".rds"))
saveRDS(elapsed_time, paste0("estData/butlerZ_time_", run, ".rds"))

# preference estimation result
e_diff = mean(abs(exp_e(theta) - exp_e(thetaEst)))
write.csv(e_diff, paste('estData/butlerZ_Eerr_', run, '.csv', sep=''))


# ---------- Our approach -------------

cat("LUQ-Learning\n")
res = piK(x1, w1, a1, y, thetaEst)
pi1_map = res$pi1_map

newResults = evaluateValue(
  pi1_map, n, p, seed, run, 
  alpha0, alpha1, beta0, beta1, delta0, delta1, sigma)

# policy evaluation result
mat = rbind(initResults, newResults)
colnames(mat) = c('E[U]', 'E[B1]', 'E[Y]')
rownames(mat) = c('observed data', 'estimated DTR'); print(mat)
write.csv(mat, paste('evaData/butlerOurs_Mat_', run, '.csv', sep=''))


# -------------- Naive --------------

cat("Naive Approach\n")
res = piK_naive(x1, w1, a1, y)
pi1_map = res$pi1_map

newResults = evaluateValue(
  pi1_map, n, p, seed, run, 
  alpha0, alpha1, beta0, beta1, delta0, delta1, sigma)

# policy evaluation result
mat = rbind(initResults, newResults)
colnames(mat) = c('E[U]', 'E[B1]', 'E[Y]')
rownames(mat) = c('observed data', 'estimated DTR'); print(mat)
write.csv(mat, paste('evaData/butlerNaive_Mat_', run, '.csv', sep=''))


# --------------- Sat ----------------

cat("Satisfaction Based Approach\n")
res = piK_sat(x1, w1, a1, as.matrix(b1), thetaEst)
pi1_map = res$pi1_map

newResults = evaluateValue(
  pi1_map, n, p, seed, run, 
  alpha0, alpha1, beta0, beta1, delta0, delta1, sigma)

# policy evaluation result
mat = rbind(initResults, newResults)
colnames(mat) = c('E[U]', 'E[B1]', 'E[Y]')
rownames(mat) = c('observed data', 'estimated DTR'); print(mat)
write.csv(mat, paste('evaData/butlerSat_Mat_', run, '.csv', sep=''))


# --------------- Known ----------------

cat("If Satisfaction Known\n")
res = piK_known(x1, w1, a1, y, as.matrix(v))
pi1_map = res$pi1_map

newResults = evaluateValue_known(
  pi1_map, n, p, seed, run, 
  alpha0, alpha1, beta0, beta1, delta0, delta1, sigma)

# policy evaluation result
mat = rbind(initResults, newResults)
colnames(mat) = c('E[U]', 'E[B1]', 'E[Y]')
rownames(mat) = c('observed data', 'estimated DTR'); print(mat)
write.csv(mat, paste('evaData/butlerKnown_Mat_', run, '.csv', sep=''))