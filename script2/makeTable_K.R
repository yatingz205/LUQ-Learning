
# Obs, Ours, Sat, Naive, Known
# load the Mat for each, over n and seed
# average over / sd over seeds to get one row of table
# rbind all the rows

setwd(dirname(rstudioapi::getSourceEditorContext()$path))

today = format(Sys.Date(), "%Y%m%d")
# N_LIST=(600 2400)
n = 600

labels = c('ours', 'sat', 'naive', 'known')
K_vec = c(2, 4, 6, 8)
seed_vec = 41 + 0:399
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

write.csv(resTable_V, paste0("./finalSummary/TableV_K_n=", n, "_", today, ".csv"))
write.csv(resTable_Vdiff, paste0("./finalSummary/TableVdiff_K_n=", n, "_", today, ".csv"))

options(width = 200)
print(resTable_V)
print(resTable_Vdiff)




# labels <- c("mis")     # already defined
# seed_vec, numSeeds     # already defined
# K_vec, numK            # define these for your K grid
# n_fixed                # pick the n you want (since we're "by K" now)
# lab, today             # already defined

n_fixed <- 2400
resTable_mae <- matrix(NA_real_, nrow = length(K_vec), ncol = length(labels) * 2)
colnames(resTable_mae) <- as.vector(t(outer(labels, c("_mean", "_sd"), paste0)))

library(reticulate)
use_condaenv("luql_env", required = TRUE)
np <- import("numpy")
numSeeds <- length(seed_vec)
numK <- length(K_vec)

# Preallocate
resMat <- matrix(NA_real_, numSeeds, numK)

for (i in seq_along(seed_vec)) {
  seed <- seed_vec[i]
  for (j in seq_along(K_vec)) {
    run <- sprintf("seed=%s_K=%s_n=%s", seed, K_vec[j], n_fixed)
    f <- file.path("estData", paste0("errors_", run, ".npy"))
    if (file.exists(f)) {
      resMat[i, j] <- as.numeric(np$load(f)[2])  # MAE (2nd entry)
    }
  }
}
out <- data.frame(
  K = K_vec,
  mae_mean = colMeans(resMat, na.rm = TRUE),
  mae_sd   = apply(resMat, 2, sd, na.rm = TRUE),
  n_found  = colSums(!is.na(resMat))
)

print(out)
write.csv(
  out, sprintf("finalSummary/TableMAE_K_n=%s_%s.csv",
  n_fixed, format(Sys.Date(), "%Y%m%d")), row.names = FALSE
)