
# Obs, Ours, Sat, Naive, Known
# load the Mat for each, over n and seed
# average over / sd over seeds to get one row of table
# rbind all the rows

setwd(dirname(rstudioapi::getSourceEditorContext()$path))

today = format(Sys.Date(), "%Y%m%d")
n = 2500

labels = c('ours', 'sat', 'naive', 'known')
K_vec = c(2, 4, 6, 8, 10)
seed_vec = 41 + 1:10
numSeeds = length(seed_vec); numK = length(K_vec)


# ---------------------------
#        Main Script 
# ---------------------------

resTable_V = matrix(NA_real_, nrow=length(K_vec), ncol=length(labels)*2)
resTable_Vdiff = matrix(NA_real_, nrow=length(K_vec), ncol=length(labels)*2)
colnames(resTable_V) = colnames(resTable_Vdiff) = as.vector(t(outer(labels, c("_mean", "_sd"), paste0)))


for(l in 1:length(labels)) {
  label = labels[l]
  resMat_V = resMat_Vdiff = matrix(NA_real_, nrow=numSeeds, ncol=numK)
  for (i in 1:numSeeds) {
    for (j in 1:numK) {
      seed = seed_vec[i]; K = K_vec[j]
      run = paste0('seed=', seed, '_K=', K, '_n=', n)
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

resTable_V = cbind(K = K_vec, resTable_V)
resTable_Vdiff = cbind(K = K_vec, resTable_Vdiff)

options(width = 200)
print(resTable_V)
print(resTable_Vdiff)

write.csv(resTable_V, paste0('evaData/TableV_K_n=',n , "_", today, '.csv'), row.names = FALSE)
write.csv(resTable_Vdiff, paste0('evaData/TableVdiff_K_n=',n , "_", today, '.csv'), row.names = FALSE)
