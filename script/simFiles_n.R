#call-in helper functions
source('helperFuncs.R')

#simulation setup
args = commandArgs(trailingOnly=TRUE)
seed = ifelse(length(args)>0, as.numeric(args[1]), 42)
nSim = ifelse(length(args)>1, as.numeric(args[2]), 1000)
n = ifelse(length(args)>2, as.numeric(args[3]), 600)
lab = ifelse(length(args)>3, as.character(args[4]), 'op')
run = paste(lab, '_seed=', seed, '_n=', n, sep='')
set.seed(seed)

#simulating variables
beta1 = matrix(rnorm(24), nrow=2)
beta2 = sqrt(0.8)*beta1 + sqrt(0.2)*array(rnorm(prod(dim(beta1))), dim=dim(beta1))
beta01 = beta02 = rep(0, 12)

alpha1 = 0.5; alpha01 = 0.75*seq(1,6)
alpha2 = 0.6; alpha02 = alpha01+0.5

gamma01 = array(rnorm(12,sd=0.5),dim=c(4,3))
gamma11 = array(rnorm(12,sd=1),dim=c(4,3))
gamma02 = array(rnorm(12,sd=1),dim=c(4,3))
gamma12 = array(0,dim=c(4,3))
gamma02[,1] = rep(0,4)
gamma02[,3] = -gamma02[,2]
#gamma02 = array(0,dim=c(4,3))
#gamma12 = array(0,dim=c(4,3))

lambda1 = 0.5; lambda2 = 2
allPerms = matrix(c(1,2,0,2,1,0,0,1,2,2,0,1,0,2,1,1,0,2), ncol=3, byrow=T)+1

#generating neccesary storage objects
w1 = w2 = matrix(nrow=n, ncol=12)
x2 = y = w1R = w2R = matrix(nrow=n, ncol=3)
b1probs = b2probs = matrix(nrow=n, ncol=7); b1 = b2 = rep(0, n)
tau_dists = matrix(nrow=n, ncol=nrow(allPerms))


#simulating trajectories (true e)
if(lab == 'op') {
    v = matrix(rnorm(2*n), ncol=2)
    v_extend = cbind(rep(1,n), v)
    e = t(apply(v, 1, softmax))
}
if(lab == 'mis') {
    v_extend = e = simulate_dirichlet(n, c(1,1,1))
}
eR = t(apply(e, 1, order))
for (j in 1:ncol(w1)) w1[,j] = rbinom(n, 1, sigmoid(v_extend %*% c(beta01[j], beta1[,j])))
x1 =  matrix(rbinom(3*n, 10, 0.5), ncol=3)
a1_sparse = sample(c(1,2,3,4), n, replace = T)
a1 = model.matrix(~-1+as.factor(a1_sparse))
for (j in 1:ncol(w2)) w2[,j] = rbinom(n, 1, sigmoid(v_extend %*% c(beta02[j], beta2[,j])))
for (j in 1:ncol(x2)) x2[,j] = rbinom(n, 10, sigmoid(gamma01[,j]%*%t(a1)+((x1[,j]-5)/2.5)*gamma11[,j]%*%t(a1)))
for (k in 2:(ncol(b1probs)-1)) b1probs[,k] = sigmoid(alpha01[k] - alpha1*rowSums(x2*e))-sigmoid(alpha01[k-1] - alpha1*rowSums(x2*e))
b1probs[,1] = sigmoid(alpha01[1] - alpha1*rowSums(x2*e)); b1probs[,7] = 1-rowSums(b1probs[,1:6])
b1 = glmnet::rmult(b1probs)
#c = sample(c(1,2,3,4), n, replace=T)
#q = quantile(rowSums(x1), probs = c(0.25, 0.5, 0.75))
#c = as.numeric(as.character(cut(rowSums(x1), breaks = c(-Inf, q[1], q[2], q[3], Inf), labels = 1:4, right = TRUE)))
# Fixed breakpoints instead of quantiles
fixed_breaks <- c(-Inf, 13, 15, 17, Inf)
c = as.numeric(as.character(cut(rowSums(x1), breaks = fixed_breaks, labels = 1:4, right = TRUE)))
new_a_sparse = sapply(1:n, function(i) sample(setdiff(seq(1,4), a1_sparse[i]), 1))
new_a = model.matrix(~-1+as.factor(new_a_sparse))
binaryDraws = rbinom(n, 1, 0.5)
a2 = a1 + new_a*(c==2 | a1[,1]==1 | (c==3 & binaryDraws==1)) - a1*((c==4 | (c==3 & binaryDraws==0)) & a1[,1]==1)
shift_x2 = apply(x2, 2, mean); scale_x2 = apply(x2, 2, sd)
for (j in 1:ncol(y)) y[,j] = rbinom(n, 10, sigmoid(gamma02[,j]%*%t(a2)+((x2[,j]-shift_x2[j])/scale_x2[j])*gamma12[,j]%*%t(a2)))
for (k in 2:(ncol(b2probs)-1)) b2probs[,k] = sigmoid(alpha02[k] - alpha2*rowSums(y*e))-sigmoid(alpha02[k-1] - alpha2*rowSums(y*e))
b2probs[,1] = sigmoid(alpha02[1] - alpha2*rowSums(y*e)); b2probs[,7] = 1-rowSums(b2probs[,1:6])
b2 = glmnet::rmult(b2probs)
for (j in 1:ncol(tau_dists)) tau_dists[,j] = apply(eR, 1, tau_metric, pi=allPerms[j,])
w1R_probs = exp(-lambda1*tau_dists) / rowSums(exp(-lambda1*tau_dists))
for (i in 1:nrow(w1R)) w1R[i,] = allPerms[sample(x = seq(1,6), size=1, prob = w1R_probs[i,]),]
w2R_probs = exp(-lambda2*tau_dists) / rowSums(exp(-lambda2*tau_dists))
for (i in 1:nrow(w2R)) w2R[i,] = allPerms[sample(x = seq(1,6), size=1, prob = w2R_probs[i,]),]


#simulating V, E (from assumed model)
if(lab == 'op' || lab == 'mis') {
    vSim = matrix(rnorm(2*nSim), nrow=nSim, ncol=2)
    vextendSim = cbind(rep(1,nSim), vSim)
    eSim = t(apply(vSim, 1, softmax))
}


#saving files to python
library(reticulate)
np = import("numpy")
makeTitle = function(objName) paste('simData/', objName, '_', run, '.npy', sep='')
np$save(makeTitle('x1'), x1)
np$save(makeTitle('v_extend'), v_extend)
np$save(makeTitle('vextendSim'), vextendSim)
np$save(makeTitle('e'), e)
np$save(makeTitle('eSim'), eSim)
np$save(makeTitle('beta01'), beta01); np$save(makeTitle('beta1'), beta1)
np$save(makeTitle('beta02'), beta02); np$save(makeTitle('beta2'), beta2)
np$save(makeTitle('alpha01'), alpha01); np$save(makeTitle('alpha1'), alpha1)
np$save(makeTitle('alpha02'), alpha02); np$save(makeTitle('alpha2'), alpha2)
np$save(makeTitle('lambda1'), lambda1); np$save(makeTitle('lambda2'), lambda2)
np$save(makeTitle('gamma01'), gamma01); np$save(makeTitle('gamma11'), gamma11)
np$save(makeTitle('gamma02'), gamma02); np$save(makeTitle('gamma12'), gamma12)
np$save(makeTitle('shift_x2'), shift_x2); np$save(makeTitle('scale_x2'), scale_x2)
np$save(makeTitle('w1'), w1); np$save(makeTitle('w2'), w2)
np$save(makeTitle('b1'), b1); np$save(makeTitle('b2'), b2)
np$save(makeTitle('a1'), a1); np$save(makeTitle('a2'), a2)
np$save(makeTitle('x2'), x2); np$save(makeTitle('y'), y)
np$save(makeTitle('w1R'), w1R); np$save(makeTitle('w2R'), w2R)
