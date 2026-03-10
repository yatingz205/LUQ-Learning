# Obs, Ours, Sat, Naive, Known
# load the Mat for each, over n and seed
# average over / sd over seeds to get one row of table
# rbind all the rows

setwd(dirname(rstudioapi::getSourceEditorContext()$path))

library(fs)
library(dplyr)

today = format(Sys.Date(), "%Y%m%d")


# ------------------------------------
#      Job status check, by n case 
# ------------------------------------

lab = 'mis'
labels = c('ours', 'sat', 'naive', 'known')
n_vec = c(300, 600, 1200, 2400, 4800)
seed_vec = 41 + 0:399
numSeeds = length(seed_vec); numN = length(n_vec)


# Build all file paths at once
runs <- expand.grid(seed = seed_vec, n = n_vec)
csv_files <- file.path("evaData", paste0("satMat_", lab, "_seed=", runs$seed, "_n=", runs$n, ".csv"))
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

check <- status %>% filter(!csv_exists) %>% 
  group_by(seed, n) %>% summarise(count = n())
print(check, n = 30)

summary(check$count)



# ------------------------------------
#      Job status check, by K case 
# ------------------------------------

n_vec = c(600, 2400)
seed_vec = 41 + 0:399
K_vec <- c(2, 4, 6, 8)

numSeeds = length(seed_vec); numN = length(n_vec)


# Build all combinations
runs <- expand.grid(seed = seed_vec, n = n_vec, K = K_vec)

# --- evaData checks ---
eva_files <- file.path(
    "evaData", paste0("satMat_", "seed=", runs$seed, "_K=", runs$K, "_n=", runs$n, ".csv")
)
eva_exists <- file.exists(eva_files)
eva_size <- ifelse(eva_exists, file.info(eva_files)$size, 0)

# --- estData checks ---
est_files <- cbind(
  time   = file.path("estData", paste0("time_",   "seed=", runs$seed, "_K=", runs$K, "_n=", runs$n, ".npy"))
)
est_exists <- apply(est_files, 2, file.exists)
est_size <- ifelse(est_exists, file.info(est_files)$size, 0)

# --- Combine into status table ---
status <- data.frame(
  seed = runs$seed,
  n = runs$n,
  K = runs$K,
  eva_file = eva_files,
  eva_exists = eva_exists,
  eva_size = eva_size,
  time_exists = est_exists[, "time"]
)

# --- Filter missing or empty ---
missing <- status %>% 
  filter(!time_exists) %>% 
  group_by(n, K, seed) %>% 
  summarise(missing_count = n(), .groups = "drop")

print(missing, n = 50)


# ------------------------------------
#      Job status check, Butler
# ------------------------------------

n_vec = c(100, 200, 500, 1500)
seed_vec = 41 + 0:399

status_butler <- expand.grid(seed = seed_vec, n = n_vec) %>%
  mutate(
    eval_file = file.path("evaData", paste0("butlerButler_Mat_seed=", seed, "_n=", n, ".csv")),
    param_file = file.path("estData", paste0("butler_param_seed=", seed, "_n=", n, ".RData")),
    error_file = file.path("estData", paste0("butler_errors_seed=", seed, "_n=", n, ".rds")),
    Eerr_file  = file.path("estData", paste0("butler_Eerr_seed=", seed, "_n=", n, ".csv")),
    time_file  = file.path("estData", paste0("butler_time_seed=", seed, "_n=", n, ".rds"))
  ) %>%
  rowwise() %>%
  mutate(
    eval_exists  = fs::file_exists(eval_file),
    param_exists = fs::file_exists(param_file),
    error_exists = fs::file_exists(error_file),
    Eerr_exists  = fs::file_exists(Eerr_file),
    time_exists  = fs::file_exists(time_file),
    eval_size    = ifelse(eval_exists, fs::file_info(eval_file)$size, 0)
  ) %>%
  ungroup() %>%
  select(-eval_file, -param_file, -error_file, -Eerr_file, -time_file)


missing <- status_butler %>% filter(!time_exists)
missing <- status_butler %>% filter(!eval_exists)
print(head(missing, 100), n = 100)


# ------------------------------------
#      Job status check, Butler_Z
# ------------------------------------

n_vec = c(100, 200, 500, 1500)
seed_vec = 41 + 0:399

status_butlerZ <- expand.grid(seed = seed_vec, n = n_vec) %>%
  mutate(
    eval_file = file.path("evaData", paste0("butlerOurs_Mat_seed=", seed, "_n=", n, ".csv")),
    param_file = file.path("estData", paste0("butlerZ_param_seed=", seed, "_n=", n, ".RData")),
    error_file = file.path("estData", paste0("butlerZ_errors_seed=", seed, "_n=", n, ".rds")),
    Eerr_file  = file.path("estData", paste0("butlerZ_Eerr_seed=", seed, "_n=", n, ".csv")),
    time_file  = file.path("estData", paste0("butlerZ_time_seed=", seed, "_n=", n, ".rds"))
  ) %>%
  rowwise() %>%
  mutate(
    eval_exists  = fs::file_exists(eval_file),
    param_exists = fs::file_exists(param_file),
    error_exists = fs::file_exists(error_file),
    Eerr_exists  = fs::file_exists(Eerr_file),
    time_exists  = fs::file_exists(time_file),
    eval_size    = ifelse(eval_exists, fs::file_info(eval_file)$size, 0)
  ) %>%
  ungroup() %>%
  select(-eval_file, -param_file, -error_file, -Eerr_file, -time_file)

missing <- status_butlerZ %>% filter(!time_exists)
missing <- status_butlerZ %>% filter(!eval_exists)
print(head(missing, 20), width = Inf)


# ===============
seed <- 140
n <- 1500
lab <- "mis"
prefix <- "satMat"  # or whichever prefix

run <- paste0(lab, "_seed=", seed, "_n=", n)
err_file <- paste0("logs_eval/", prefix, "_", run, ".err")
err_file

cat(readLines(err_file), sep = "\n")