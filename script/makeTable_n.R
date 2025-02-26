
# Obs, Ours, Sat, Naive, Known
# load the Mat for each, over n and seed
# average over / sd over seeds to get one row of table
# rbind all the rows

setwd(dirname(rstudioapi::getSourceEditorContext()$path))

butler = FALSE
sn = 2 # 1 or 2, specify only for butler = FALSE
today = format(Sys.Date(), "%Y%m%d")

if(!butler){
  labels = c('ours', 'sat', 'naive', 'known')
  n_vec = c(150, 300, 600, 1200, 2500, 6000)
  lab = paste0('_sn=', sn)
} else {
  labels = c('butlerOurs_', 'butlerSat_', 'butlerNaive_', 'butlerKnown_', 'butlerButler_')
  n_vec = c(100, 200, 500, 1000)  
  lab = paste0('_butler')
}

seed_vec = 41 + 1:10
numSeeds = length(seed_vec); numN = length(n_vec)


# ---------------------------
#        Main Script 
# ---------------------------

resTable_V = matrix(NA_real_, nrow=length(n_vec), ncol=length(labels)*2)
resTable_Vdiff = matrix(NA_real_, nrow=length(n_vec), ncol=length(labels)*2)
colnames(resTable_V) = colnames(resTable_Vdiff) = as.vector(t(outer(labels, c("_mean", "_sd"), paste0)))


for(l in 1:length(labels)) {
  label = labels[l]
  resMat_V = resMat_Vdiff = matrix(NA_real_, nrow=numSeeds, ncol=numN)
  for (i in 1:numSeeds) {
    for (j in 1:numN) {
      seed = seed_vec[i]; n = n_vec[j]
      run = paste0('sn=', sn, '_seed=', seed, '_n=', n)
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
  resTable_V[,c(2*(l-1)+1,2*(l-1)+2)] = cbind(mean_V, sd_V)
  resTable_Vdiff[,c(2*(l-1)+1,2*(l-1)+2)] = cbind(mean_Vdiff, sd_Vdiff)
}

resTable_V = cbind(N = n_vec, resTable_V)
resTable_Vdiff = cbind(N = n_vec, resTable_Vdiff)

options(width = 200)
print(resTable_V)
print(resTable_Vdiff)

write.csv(resTable_V, paste0('evaData/TableV_n', lab, "_", today, '.csv'), row.names = FALSE)
write.csv(resTable_Vdiff, paste0('evaData/TableVdiff_n', lab, "_", today, '.csv'), row.names = FALSE)
