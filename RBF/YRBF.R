# Função que calcula a saı́da de uma rede RBF.
library("corpcor")

YRBF <- function(xin, modRBF){
  ####### Função radial Gaussiana ########
  pdfnvar<-function(x,m,K,n){
    if (n==1) {
      r<-sqrt(as.numeric(K))
      px<-(1/(sqrt(2*pi*r*r))) * exp(-0.5 *((x-m)/r)^2)
    }
    else {
      px<-((1/(sqrt((2*pi)^n * (det(K))))) * exp (-0.5 * (t(x-m) %*% (solve(K)) %*% (x-m))))
    }
  }
  ########################################
  N <- dim(xin)[1] # número de amostras
  n <- dim(xin)[2] # dimensão de entrada (deve ser maior que 1)
  m <- as.matrix(modRBF[[1]])
  covlist <- modRBF[[2]]
  p <- length(covlist) # Número de funções radiais
  W <- modRBF [[3]]
  
  xin <- as.matrix(xin) # garante que xin seja matriz
  
  H <- matrix(nrow = N, ncol = p)
  # Calcula matriz H
  for (j in 1:N) {
    for (i in 1:p) {
      mi <- m[i, ]
      covi <- covlist[i]
      covi <- matrix(unlist(covlist[i]), ncol = n, byrow = T) + 0.001 * diag(n)
      H[j,i] <- pdfnvar(xin[j, ], mi, covi, n)
    }
  }
  
  Haug <- cbind(1, H)
  Yhat <- Haug %*% W
  return(Yhat)
}