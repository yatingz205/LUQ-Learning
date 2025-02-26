#define helper functions
sigmoid = function(x) 1/(1+exp(-x))
softmax = function(x) c(exp(x),1)/(sum(c(exp(x),1)))
advancedIndex = function(mat,vec) mat[cbind(seq_len(nrow(mat)), vec)]

tau_metric = function(pi0, pi) {
  tau = 0
  for (i in 2:3) {
    for (j in 1:(i-1)) {
      tau = tau + ((pi0[i]-pi0[j])*(pi[i]-pi[j])<0)
    }
  }
  return(tau)
}

ensure_file_exists <- function(filename, script_to_run, args = "") {
  if (!file.exists(filename)) {
    cat("File", filename, "not found. Running", script_to_run, "with arguments:", args, "\n")
    system(paste("Rscript", script_to_run, args))  
  }
}
