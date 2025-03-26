# script to generate plot of error on theta and computation time
setwd(dirname(rstudioapi::getSourceEditorContext()$path))

library(reticulate)
use_condaenv('prl_env')
library(ggplot2)
np = import("numpy")

n = 2500 # 600 or 2500
today = format(Sys.Date(), "%Y%m%d")


# -------------------------
#           Setup
# -------------------------
K_vec = c(2, 4, 6, 8, 10)
seed_vec = 41 + 1:10
numSeeds = length(seed_vec); numK = length(K_vec)
timeMat = errMat = matrix(nrow=numSeeds, ncol=numK)
colnames(timeMat) = colnames(errMat) = K_vec
rownames(timeMat) = rownames(errMat) = seed_vec


# -------------------------
#        Gather Data
# -------------------------
loadTheta = function(run) {
  #load true values
  makeTitle = function(objName) paste('simData/', objName, '_', run, '.npy', sep='')
  beta1 = np$load(makeTitle('beta0')); beta2 = np$load(makeTitle('beta1'))
  alpha1 = as.vector(np$load(makeTitle('alpha0'))); alpha2 = as.vector(np$load(makeTitle('alpha1')))
  
  #load estimates
  makeTitle = function(objName) paste('estData/', objName, '_', run, '.npy', sep='')
  beta_opt = np$load(makeTitle('beta_opt'))
  alpha_opt = np$load(makeTitle('alpha_opt'))
  alpha1_opt = as.vector(alpha_opt[1:(nrow(alpha_opt)-1), , drop = FALSE])
  alpha2_opt = as.vector(alpha_opt[nrow(alpha_opt), , drop = FALSE])
  beta1_opt = beta_opt[, , 1, drop=FALSE]
  beta2_opt = beta_opt[, , 2:dim(beta_opt)[3], drop=FALSE]
  
  #concatenate parms and return
  theta = c(alpha1, alpha2, beta1, beta2)
  theta_opt = c(alpha1_opt, alpha2_opt, beta1_opt, beta2_opt)
  return(list(theta=theta, theta_opt=theta_opt, dim=length(theta)))
}


#populate matrices
for (i in 1:numSeeds) {
  for (j in 1:numK) {
    seed = seed_vec[i]; K = K_vec[j]
    run = paste('seed=', seed, '_Ksm=', K, '_n=', n, sep='')
    file_name = paste0('estData/time_', run, '.npy')
    if(file.exists(file_name)) timeMat[i,j] = np$load(file_name)
    file_name = paste0('estData/errors_', run, '.npy')
    if(file.exists(file_name)) errMat[i,j] = np$load(file_name)[2]
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
  K = K_vec
)


comp_plot = ggplot(data=plot_data, mapping = aes(x=K)) +
  geom_errorbar(aes(ymin=pmax(0,time_mean-time_sd), ymax=time_mean+time_sd), width=0.2, na.rm=TRUE) +
  geom_point(aes(y=time_mean), na.rm=TRUE) + 
  geom_line(aes(y=time_mean), na.rm=TRUE) +
  xlab('K') + 
  ylab("Computation Time (minutes)") +
  theme_minimal(base_size = 15) + 
  theme(panel.grid.minor = element_blank()) 

pdf(file=paste0("evaData/plot_comptime_K_n=",n,"_", today, ".pdf"), width=7, height=4)
plot(comp_plot)
dev.off()


error_plot = ggplot(data=plot_data, mapping = aes(x=K)) +
  geom_errorbar(aes(ymin=pmax(0,err_mean-err_sd), ymax=err_mean+err_sd), width=0.2, na.rm=TRUE) +
  geom_point(aes(y=err_mean), na.rm=TRUE) + 
  geom_line(aes(y=err_mean), na.rm=TRUE) +
  xlab('K') + 
  ylab('Mean Absolute Error') + 
  theme_minimal(base_size = 15) +
  theme(panel.grid.minor = element_blank()) 

pdf(file=paste0("evaData/plot_err_K_n=",n,"_", today, ".pdf"), width=7, height=4)
plot(error_plot)
dev.off()


