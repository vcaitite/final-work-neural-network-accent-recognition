library("corpcor")

trainELM <- function(xin, yin, p, par){
  n <- dim(xin) [2] # Dimensão da entrada

  #Adiciona ou não o termo de polarização
  if(par == 1){
    xin<-cbind(1,xin)
    Z<-replicate(p, runif(n+1, -0.5, 0.5))
  }
  else{
    Z<-replicate(p, runif(n, -0.5, 0.5))
  }
  H<-tanh(xin %*% Z)
  
  W<-pseudoinverse(H)%*%yin
  #W<-(solve(t(H) %*% H) %*% t(H)) %*% yin
  
  return(list(W,H,Z))
}