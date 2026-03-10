args = commandArgs(trailingOnly=TRUE)

suppressMessages({
  library(doParallel)
  library(reticulate)
  library(caret)
})

seed = ifelse(length(args)>0, as.numeric(args[1]), 42)
n = ifelse(length(args)>1, as.numeric(args[2]), 600)
lab = ifelse(length(args)>2, as.character(args[3]), 'op')
run = paste(lab, '_seed=', seed, '_n=', n, sep='')

#output_file = paste('evaData/knownMat_', run, '.csv', sep='')
#if (file.exists(output_file)) {
#  message("File ", output_file, " Already exists. Exiting script.")
#  quit(save = "no", status = 0)
#}

seed = as.numeric(seed)
dir.create("evaData", showWarnings = FALSE, recursive = TRUE)
n = as.numeric(n)

set.seed(seed)
print(run)


use_condaenv("luql_env", required = TRUE)
np = import("numpy")
allPerms = matrix(c(1,2,0,2,1,0,0,1,2,2,0,1,0,2,1,1,0,2), ncol=3, byrow=T)+1
source('helperFuncs.R')

#load training data
makeTitle = function(objName) paste('simData/', objName, '_', run, '.npy', sep='')
v_extend = np$load(makeTitle('v_extend')); vextendSim = np$load(makeTitle('v_extended_sim'))
e = np$load(makeTitle('e')); eSim = np$load(makeTitle('e_sim'))
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



# --------------------------------
#         Model Evaluation
# --------------------------------

# load functions and simulated data
source('genNamespace_n.R')

# observational value
initResults = c(mean(rowSums(y*e)), mean(b2), mean(y))

# value of DTR (on testing set), n testing set to be the same as n training
newResults = evaluateValue(
  # Training data
  x1_train = x1, x2_train = x2, 
  a1_train = a1, a2_train = a2, b1_train = b1, b2_train = b2,
  w1_train = w1, w2_train = w2, w1R_train = w1R, w2R_train = w2R, 
  y_train = y,
  # Test data parameters
  n_test = n, test_seed = seed + 1000, lab = lab,
  # True model parameters
  alpha01, alpha1, alpha02, alpha2,
  beta01, beta1, beta02, beta2,
  lambda1, lambda2, gamma01, gamma11, gamma02, gamma12,
  # Estimated parameters
  alpha01_opt, alpha1_opt, alpha02_opt, alpha2_opt,
  beta01_opt, beta1_opt, beta02_opt, beta2_opt,
  lambda1_opt, lambda2_opt,
  # Additional needed parameters
  vextendSim = vextendSim, eSim = eSim, 
  allPerms = allPerms, e_train = e, 
  type = 'known'
)

mat = rbind(initResults, newResults)
colnames(mat) = c('E[U]', 'E[B2]', 'E[Y]')
rownames(mat) = c('observed data', 'estimated DTR')
print(mat)

write.csv(mat, paste('evaData/knownMat_', run, '.csv', sep=''))

closeAllConnections()
