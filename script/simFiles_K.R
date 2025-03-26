#call-in helper functions
source('helperFuncs.R')

#simulating parameters
args = commandArgs(trailingOnly = TRUE)
seed = ifelse(length(args)>0, as.double(args[1]), 42)
K = ifelse(length(args)>1, as.double(args[2]), 2)
n = ifelse(length(args)>2, as.double(args[3]), 600)
nSim = ifelse(length(args)>3, as.double(args[4]), 1000)
set.seed(seed)

#generating model parameters
num_w = 2; num_v = 2; num_b = 3
num_y = num_v + 1; num_alpha = num_b - 1

beta0 = matrix(0, nrow=num_w, ncol=K)
beta1 = array(0, dim=c(num_w, K, num_v))
beta1[,1,] = rnorm(num_w * num_v)
for (k in 1:(K-1)) beta1[,k+1,] = sqrt(0.8)*beta1[,k,] + sqrt(0.2)*rnorm(num_w * num_v)

alpha1 = 0.6 + 0.05*c(1:K) - 0.1*K
alpha0 = matrix(0, nrow=num_alpha, ncol=K)
alpha0[,1] = 0.75*c(1:num_alpha)
for (k in 1:(K-1)) alpha0[,k+1] = alpha0[,1] + k/(4*(K-1))

gamma0 = gamma1 = array(0, dim=c(4,num_v+1,K))
gamma0[,,1] = rnorm(4*num_y, 0, 0.5)
gamma1[,,1] = rnorm(4*num_y, 0, 1)
for (k in 1:(K-1)) {
  gamma0[,,k+1] = sqrt(0.8)*gamma0[,,k] + sqrt(0.2)*rnorm(4*num_y, sd=0.5)
  gamma1[,,k+1] = sqrt(0.8)*gamma1[,,k] + sqrt(0.2)*rnorm(4*num_y)
}


#creating neccesary storage objects
W = array(dim=c(n, num_w, K))
X = array(dim=c(n, num_y, K+1))
A = array(dim=c(n, 4, K))
B = array(dim=c(n, K))
bprobs = array(dim=c(n, num_b, K))
shift_x = scale_x = matrix(nrow=3, ncol=K)


#simulating trajectories
V = matrix(rnorm(num_v*n), ncol=num_v)
E = t(apply(V, 1, softmax))
ER = t(apply(E, 1, order))
X[,,1] = rbinom(num_y*n, 10, 0.5)
A_sparse = matrix(sample(c(1,2,3,4), K*n, replace=T), ncol=K)
for (k in 1:K) {
  for (j in 1:num_w) W[,j,k] = rbinom(n, 1, sigmoid(beta0[j,k] + V%*%beta1[j,k,]))
  A[,,k] = model.matrix(~-1+as.factor(A_sparse[,k]))
  shift_x[,k] = apply(X[,,k], 2, mean); scale_x[,k] = apply(X[,,k], 2, sd)
  XS = t((t(X[,,k]) - shift_x[,k])/scale_x[,k])
  for (j in 1:num_y) {
    probs_j = rowSums(A[,,k] * t(gamma0[,j,k] + t(XS[,j,drop=F]%*%t(gamma1[,j,k]))))
    X[,j,k+1] = rbinom(n, 10, sigmoid(probs_j))
  }
  rel_odds = alpha1[k]*rowSums(X[,,k+1]*E)
  bprobs[,1,k] = sigmoid(alpha0[1,k] - rel_odds)
  for (j in 2:num_alpha) bprobs[,j,k] = sigmoid(alpha0[j,k]-rel_odds) - sigmoid(alpha0[j-1,k]-rel_odds)
  bprobs[,num_b,k] = 1 - rowSums(bprobs[,1:num_alpha,k])
  B[,k] = glmnet::rmult(bprobs[,,k])
}
y = X[,,K+1]


#simulating V from known marginal
vSim = matrix(rnorm(num_v*nSim), nrow=nSim, ncol=num_v)

#saving files to python
library(reticulate)
np = import("numpy")
run = paste('seed=', seed, '_K=', K, '_n=', n, sep='')
print(run)
makeTitle = function(objName) paste('simData/', objName, '_', run, '.npy', sep='')
np$save(makeTitle('V'), V); np$save(makeTitle('W'), W)
np$save(makeTitle('A'), A); np$save(makeTitle('X'), X)
np$save(makeTitle('y'), y); np$save(makeTitle('B'), B)
np$save(makeTitle('vSim'), vSim)
np$save(makeTitle('beta0'), beta0); np$save(makeTitle('beta1'), beta1)
np$save(makeTitle('alpha0'), alpha0); np$save(makeTitle('alpha1'), alpha1)
np$save(makeTitle('gamma0'), gamma0); np$save(makeTitle('gamma1'), gamma1)
np$save(makeTitle('shift_x'), shift_x); np$save(makeTitle('scale_x'), scale_x)
