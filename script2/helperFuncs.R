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

simulate_dirichlet <- function(nSim, alpha) {
  E <- matrix(rgamma(nSim * length(alpha), shape = alpha, rate = 1), ncol = length(alpha), byrow = TRUE)
  E <- E / rowSums(E)  
  return(E)
}
