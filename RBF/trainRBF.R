# Função de treinamento de uma rede RBF.
library("corpcor")

trainRBF <- function(xin, yin, p){
  ####### Função radial Gaussiana ########
  pdfnvar<-function(x,m,K,n){
    if (n==1) {
      r<-sqrt(as.numeric(K))
      px<-(1/(sqrt(2*pi*r*r)))*exp(-0.5 *((x-m)/r)^2)
    }
    else {
      px<-((1/(sqrt((2*pi)^n * (det(K))))) * exp (-0.5 * (t(x-m) %*% (solve(K)) %*% (x-m)))) #eq 6.5
    }
  }
  ########################################
  N<-dim(xin)[1] # número de amostras
  n<-dim(xin)[2] # dimensão de entrada (deve ser maior que 1)

  xin <- as.matrix(xin) # garante que xin seja matriz
  yin <- as.matrix(yin) # garante que yin seja matriz
  
  # Aplica o algoritmo kmeans para separar os clusters 
  xclust<-kmeans(xin, p)
  
  # Armazena vetores de centros das funções:
  m <- as.matrix (xclust$centers)
  covlist <- list()
  
  # Estima matrizes de covariância para todos os centros:
  for ( i in 1:p)
  {
    ici <- which(xclust$cluster == i )
    xci <- xin [ici, ]
    if(n==1){
      covi <- var(xci)  
    }
    else{ 
      row <- dim(xci)[1];
      if(is.null(row)){
        row <- 0
      }
      # Para garantir que não haverá erro (caso tenha apenas uma linha)
      if(row > 1){
        covi <- cov(xci)  
      }
      else{
        # cov de 2 linhas iguais que vai dar 0
        covi <- cov(matrix(c(xci, xci), nrow = 2))
      }
    }
    covlist [[i]] <- covi  
  }
  
  H <- matrix(nrow = N, ncol = p)
  # Calcula matriz H
  for (j in 1:N) {
    for (i in 1:p) {
      mi <- m[i, ]
      covi <- covlist[i]
      covi <- matrix(unlist(covlist[i]), ncol = n, byrow = T) + 0.001 * diag(n)
      H[j,i] <- pdfnvar(xin[j, ] , mi, covi, n)
    }
  }
  
  Haug <-cbind(1, H)
  W <- pseudoinverse(Haug) %*% yin
  
  return (list(m, covlist, W, H))
}