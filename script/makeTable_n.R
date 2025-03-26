
# Obs, Ours, Sat, Naive, Known
# load the Mat for each, over n and seed
# average over / sd over seeds to get one row of table
# rbind all the rows

setwd(dirname(rstudioapi::getSourceEditorContext()$path))

today = format(Sys.Date(), "%Y%m%d")

labels = c('ours', 'sat', 'naive', 'known')
n_vec = c(150, 300, 600, 1200, 2500, 6000)
seed_vec = 41 + 1:10
numSeeds = length(seed_vec); numN = length(n_vec)


# ---------------------------i
#        Main Script 
# ---------------------------
lab = 'mis'

resTable_V = matrix(NA_real_, nrow=length(n_vec), ncol=length(labels)*2)
resTable_Vdiff = matrix(NA_real_, nrow=length(n_vec), ncol=length(labels)*2)
colnames(resTable_V) = colnames(resTable_Vdiff) = as.vector(t(outer(labels, c("_mean", "_sd"), paste0)))

for(l in 1:length(labels)) {
  label = labels[l]
  resMat_V = resMat_Vdiff = matrix(NA_real_, nrow=numSeeds, ncol=numN)
  for (i in 1:numSeeds) {
    for (j in 1:numN) {
      seed = seed_vec[i]; n = n_vec[j]
      run = paste0(lab, '_seed=', seed, '_n=', n)
      file_name = paste0('evaData/', label, 'Mat_', run, '.csv')
      if(!file.exists(file_name)) next
      dat = read.csv(file_name, stringsAsFactors = FALSE)
      resMat_V[i,j] = dat[2,2]
      resMat_Vdiff[i,j] = dat[2,2] - dat[1,2]
    }
  }
  mean_V = apply(resMat_V, 2, mean, na.rm=TRUE)
  sd_V = apply(resMat_V, 2, sd, na.rm=TRUE)
  mean_Vdiff = apply(resMat_Vdiff, 2, mean, na.rm=TRUE)
  sd_Vdiff = apply(resMat_Vdiff, 2, sd, na.rm=TRUE)
  # print to check if all seeds by K have ran
  cat(label, '\n'); print(resMat_Vdiff); cat('\n\n')
  resTable_V[,c(2*(l-1)+1,2*(l-1)+2)] = cbind(mean_V, sd_V)
  resTable_Vdiff[,c(2*(l-1)+1,2*(l-1)+2)] = cbind(mean_Vdiff, sd_Vdiff)
}

resTable_V = cbind(N = n_vec, resTable_V)
resTable_Vdiff = cbind(N = n_vec, resTable_Vdiff)

options(width = 200)
print(resTable_V)
print(resTable_Vdiff)

write.csv(resTable_V, paste0('evaData/TableV_', lab, '_n_', today, '.csv'), row.names = FALSE)
write.csv(resTable_Vdiff, paste0('evaData/TableVdiff_', lab, '_n_', today, '.csv'), row.names = FALSE)



# Get E[hat E - E] table 
labels = c('mis', 'op')
resTable_E = matrix(NA_real_, nrow=length(n_vec), ncol=length(labels)*2)
colnames(resTable_E) = as.vector(t(outer(labels, c("_mean", "_sd"), paste0)))

for(l in 1:length(labels)) {
    label = labels[l]
    resMat_E = matrix(NA_real_, nrow=numSeeds, ncol=numN)
    for (i in 1:numSeeds) {
        for (j in 1:numN) {
            seed = seed_vec[i]; n = n_vec[j]
            run = paste(lab, '_seed=', seed, '_n=', n, sep=''); print(run)
            # loads true and estimated parameters
            source('genNamespace_n.R')
            Etrue = cond_exp(
                x1, x2, a1, a2, b1, b2, w1, w2, w1R, w2R, vSim, 
                alpha01, alpha1, alpha02, alpha2, 
                beta01, beta1, beta02, beta2, 
                lambda1, lambda2)
            Eest = cond_exp(
                x1, x2, a1, a2, b1, b2, w1, w2, w1R, w2R, vSim, 
                alpha01_opt, alpha1_opt, alpha02_opt, alpha2_opt, 
                beta01_opt, beta1_opt, beta02_opt, beta2_opt, 
                lambda1_opt, lambda2_opt)
            resMat_E[i,j] = mean(abs(Etrue - Eest))
        }
    }
    print(resMat_E)
    mean_E = apply(resMat_E, 2, mean, na.rm=TRUE)
    sd_E = apply(resMat_E, 2, sd, na.rm=TRUE)
    resTable_E[,c(2*(l-1)+1,2*(l-1)+2)] = cbind(mean_E, sd_E)
}
resTable_E = cbind(N = n_vec, resTable_E)
print(resTable_E)

write.csv(resTable_E, paste0('evaData/TableEerr_n_', lab, "_", today, '.csv'), row.names = FALSE)
