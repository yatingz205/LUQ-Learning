#script to gather table for CATIE trial
setwd(dirname(rstudioapi::getSourceEditorContext()$path))

today = format(Sys.Date(), "%Y%m%d")

n_vec = c(100, 200, 500, 1000)  
lab = paste0('butler')


seed_vec = 41 + 1:10
numSeeds = length(seed_vec); numN = length(n_vec)


# ----------------------------
#        Est V diff Table 
# ----------------------------

labels = c('butlerOurs', 'butlerSat', 'butlerNaive', 'butlerKnown', 'butlerButler')
resTable_V = matrix(NA_real_, nrow=length(n_vec), ncol=length(labels)*2)
resTable_Vdiff = matrix(NA_real_, nrow=length(n_vec), ncol=length(labels)*2)
colnames(resTable_V) = colnames(resTable_Vdiff) = as.vector(t(outer(labels, c("_mean", "_sd"), paste0)))


for(l in 1:length(labels)) {
  label = labels[l]
  resMat_V = resMat_Vdiff = matrix(NA_real_, nrow=numSeeds, ncol=numN)
  for (i in 1:numSeeds) {
    for (j in 1:numN) {
      seed = seed_vec[i]; n = n_vec[j]
      run = paste0('seed=', seed, '_n=', n)
      file_name = paste0('evaData/', label, '_Mat_', run, '.csv')
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

write.csv(resTable_V, paste0('evaData/TableV_n_', lab, "_", today, '.csv'), row.names = FALSE)
write.csv(resTable_Vdiff, paste0('evaData/TableVdiff_n_', lab, "_", today, '.csv'), row.names = FALSE)



# ----------------------------
#        Est E diff Table 
# ----------------------------

labels = c('butler', 'butlerZ')
resTable_E = matrix(NA_real_, nrow=length(n_vec), ncol=length(labels)*2)
colnames(resTable_E) = as.vector(t(outer(labels, c("_mean", "_sd"), paste0)))

for(l in 1:length(labels)) {
  label = labels[l]
  resMat_E = matrix(NA_real_, nrow=numSeeds, ncol=numN)
  for (i in 1:numSeeds) {
    for (j in 1:numN) {
      seed = seed_vec[i]; n = n_vec[j]
      run = paste0('seed=', seed, '_n=', n)
      file_name = paste0('estData/', label, '_Eerr_', run, '.csv')
      if(file.exists(file_name)) dat = read.csv(file_name, stringsAsFactors = FALSE); resMat_E[i,j] = dat$x
    }
  }
  # print to check if all seeds by K have ran
  cat(label, '\n'); print(resMat_E); cat('\n\n')
  mean_E = apply(resMat_E, 2, mean, na.rm=TRUE)
  sd_E = apply(resMat_E, 2, sd, na.rm=TRUE)
  resTable_E[,c(2*(l-1)+1,2*(l-1)+2)] = cbind(mean_E, sd_E)
}
resTable_E = cbind(N = n_vec, resTable_E)
print(resTable_E)

write.csv(resTable_E, paste0('evaData/TableEerr_n_', lab, "_", today, '.csv'), row.names = FALSE)
