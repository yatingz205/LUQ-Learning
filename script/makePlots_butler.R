# script to generate plot of error on theta and computation time
setwd(dirname(rstudioapi::getSourceEditorContext()$path))

library(reticulate)
use_condaenv('prl_env')
library(ggplot2)
np = import("numpy")

seed_vec = 41 + 1:10
today = format(Sys.Date(), "%Y%m%d")

labels = c('butlerOurs_', 'butlerSat_', 'butlerNaive_', 'butlerKnown_', 'butlerButler_')
n_vec = c(100, 200, 500, 1000)


# -------------------------
#           Setup
# -------------------------

numSeeds = length(seed_vec); numN = length(n_vec)
timeMat = errMat = matrix(nrow=numSeeds, ncol=numN)
colnames(timeMat) = colnames(errMat) = n_vec
rownames(timeMat) = rownames(errMat) = seed_vec


# -------------------------
#        Gather Data
# -------------------------
#function to extras true and estimated parm vector
loadTheta = function(run) {
  #load true values
  makeTitle = function(objName) paste('simData/', objName, '_', run, '.npy', sep='')
  beta01 = as.vector(np$load(makeTitle('beta01'))); beta02 = as.vector(np$load(makeTitle('beta02')))
  beta1 = np$load(makeTitle('beta1')); beta2 = np$load(makeTitle('beta2'))
  alpha01 = as.vector(np$load(makeTitle('alpha01'))); alpha02 = as.vector(np$load(makeTitle('alpha02')))
  alpha1 = as.vector(np$load(makeTitle('alpha1'))); alpha2 = as.vector(np$load(makeTitle('alpha2')))
  lambda1 = as.vector(np$load(makeTitle('lambda1'))); lambda2 = as.vector(np$load(makeTitle('lambda2')))
  gamma01 = np$load(makeTitle('gamma01')); gamma11 = np$load(makeTitle('gamma11'))
  gamma02 = np$load(makeTitle('gamma02')); gamma12 = np$load(makeTitle('gamma12'))
  
  #load estimates
  makeTitle = function(objName) paste('estData/', objName, '_', run, '.npy', sep='')
  beta_opt = np$load(makeTitle('beta_opt'))
  alpha_opt = np$load(makeTitle('alpha_opt'))
  lambda_opt = np$load(makeTitle('lambda_opt'))
  beta01_opt = beta_opt[1,]; beta1_opt = beta_opt[2:3,]; beta02_opt = beta_opt[4,]; beta2_opt = beta_opt[5:6,]
  alpha01_opt = alpha_opt[1,1:2]; alpha1_opt = alpha_opt[1,3]; alpha02_opt = alpha_opt[2,1:2]; alpha2_opt = alpha_opt[2,3]
  lambda1_opt = lambda_opt[1]; lambda2_opt = lambda_opt[2]
  
  #concatenate parms and return
  theta = c(alpha01, alpha1, alpha02, alpha2, beta01, beta1, beta02, beta2, lambda1, lambda2)
  theta_opt = c(alpha01_opt, alpha1_opt, alpha02_opt, alpha2_opt, beta01_opt, beta1_opt, beta02_opt, beta2_opt, lambda1_opt, lambda2_opt)
  return(list(theta=theta, theta_opt=theta_opt, dim = length(theta)))
}

#populate matrices
for (i in 1:numSeeds) {
  for (j in 1:numN) {
    seed = seed_vec[i]; n = n_vec[j]
    run = paste('seed=', seed, '_n=', n, sep='')
    file_name = paste0('estData/butler_time_', run, '.rds')
    if (file.exists(file_name)) timeMat[i, j] = readRDS(file_name)[1]
    file_name = paste0('estData/butler_errors_', run, '.rds')
    if (file.exists(file_name)) errMat[i, j] = readRDS(file_name)$mae
  }
}



# -------------------------
#        Make Plots
# -------------------------
#statistics for plotting
plot_data = data.frame(
  time_mean = apply(timeMat, 2, mean, na.rm = TRUE)/60, 
  time_sd = apply(timeMat, 2, sd, na.rm = TRUE)/60,
  err_mean = apply(errMat, 2, mean, na.rm = TRUE),
  err_sd = apply(errMat, 2, sd, na.rm = TRUE),
  n = n_vec
)


comp_plot = ggplot(data=plot_data, mapping = aes(x=n)) +
  geom_errorbar(aes(ymin=pmax(0,time_mean-time_sd), ymax=time_mean+time_sd), width=0.2, na.rm=TRUE) +
  geom_point(aes(y=time_mean), na.rm=TRUE) + 
  geom_line(aes(y=time_mean), na.rm=TRUE) +
  scale_x_log10(breaks=plot_data$n) + 
  xlab('Sample Size') + 
  ylab('Computation Time (minutes)') + 
  theme_minimal(base_size = 15) + 
  theme(panel.grid.minor = element_blank()) 

if(butler) {filename = paste0("evaData/plot_comptime_n_butler_", today, ".pdf")} else {
filename = paste0("evaData/plot_comptime_n_butler_", today, ".pdf")}
pdf(file= filename, width=8, height=4)
plot(comp_plot)
dev.off()


error_plot = ggplot(data=plot_data, mapping = aes(x=n)) +
  geom_errorbar(aes(ymin=pmax(0,err_mean-err_sd), ymax=err_mean+err_sd), width=0.2, na.rm=TRUE) +
  geom_point(aes(y=err_mean), na.rm=TRUE) + 
  geom_line(aes(y=err_mean), na.rm=TRUE) +
  scale_x_log10(breaks=plot_data$n) + 
  xlab('Sample Size') + 
  ylab('Mean Absolute Error') + 
  theme_minimal(base_size = 15) +
  theme(panel.grid.minor = element_blank()) 

if(butler) {filename = paste0("evaData/plot_err_n_butler_", today, ".pdf")} else {
filename = paste0("evaData/plot_err_n_butler_", today, ".pdf")}
pdf(file=filename, width=8, height=4)
plot(error_plot)
dev.off()


