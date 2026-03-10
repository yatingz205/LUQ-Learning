
# Obs, Ours, Sat, Naive, Known
# load the Mat for each, over n and seed
# average over / sd over seeds to get one row of table
# rbind all the rows

setwd(dirname(rstudioapi::getSourceEditorContext()$path))

library(dplyr)
library(reticulate)

today = format(Sys.Date(), "%Y%m%d")

labels = c('ours', 'sat', 'naive', 'known')
n_vec = c(300, 600, 1200, 2400, 4800)
seed_vec = 41 + 0:399
numSeeds = length(seed_vec); numN = length(n_vec)



# ---------------------------
#      Job status check 
# ---------------------------

lab = 'mis'

# Build all file paths at once
runs <- expand.grid(seed = seed_vec, n = n_vec)
csv_files <- file.path("evaData", paste0("knownMat_", lab, "_seed=", runs$seed, "_n=", runs$n, ".csv"))
npy_files <- file.path("estData", paste0("time_", lab, "_seed=", runs$seed, "_n=", runs$n, ".npy"))

# Vectorized existence check
csv_exists <- file.exists(csv_files)
npy_exists <- file.exists(npy_files)

# Vectorized size check
csv_size <- ifelse(csv_exists, file.info(csv_files)$size, 0)
npy_size <- ifelse(npy_exists, file.info(npy_files)$size, 0)

status <- data.frame(
  seed = runs$seed,
  n = runs$n,
  csv_file = csv_files,
  csv_exists = csv_exists,
  csv_size = csv_size,
  npy_file = npy_files,
  npy_exists = npy_exists,
  npy_size = npy_size
)

check <- status %>% filter(!npy_exists) %>% 
  group_by(seed, n) %>% summarise(count = n())
print(check, n = Inf)


# ---------------------------
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

write.csv(resTable_V, paste0('finalSummary/TableV_', lab, '_n_', today, '.csv'), row.names = FALSE)
write.csv(resTable_Vdiff, paste0('finalSummary/TableVdiff_', lab, '_n_', today, '.csv'), row.names = FALSE)



# Get E[hat E - E] table 
# labels = c('mis', 'op')
lab <- "op"
resTable_mae <- matrix(NA_real_, nrow = length(n_vec), ncol = length(labels) * 2)
colnames(resTable_mae) <- as.vector(t(outer(labels, c("_mean", "_sd"), paste0)))

use_condaenv("luql_env", required = TRUE)
np <- import("numpy")

for (l in seq_along(labels)) {
  label <- labels[l]
  resMat_mae <- matrix(NA_real_, nrow = numSeeds, ncol = numN)
  for (i in seq_len(numSeeds)) {
    for (j in seq_len(numN)) {
      seed <- seed_vec[i]
      n <- n_vec[j]
      run <- paste0(lab, "_seed=", seed, "_n=", n)
      f_err <- file.path("estData", paste0("errors_", run, ".npy"))
      if (file.exists(f_err)) {
        err_vec <- np$load(f_err)     # length 3: (lInf_grads, mae_error, lInf_error)
        resMat_mae[i, j] <- as.numeric(err_vec[2]) 
      } else {
        resMat_mae[i, j] <- NA_real_
      }
    }
  }
  mean_mae <- apply(resMat_mae, 2, mean, na.rm = TRUE)
  sd_mae   <- apply(resMat_mae, 2, sd,   na.rm = TRUE)

  resTable_mae[, c(2*(l-1)+1, 2*(l-1)+2)] <- cbind(mean_mae, sd_mae)
}

resTable_mae <- cbind(N = n_vec, resTable_mae)
print(resTable_mae)

write.csv(
  resTable_mae,
  paste0("finalSummary/TableMAE_n_", lab, "_", today, ".csv"),
  row.names = FALSE
)

