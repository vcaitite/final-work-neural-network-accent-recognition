YELM<-function(xin, Z, W, par){
  n<-dim(xin)[2]
  
  # Adiciona ou não termo de polarização
  if(par == 1) {
    xin<-cbind(1, xin) 
  }
  H<-tanh(xin%*%Z)
  y_hat<-sign(H %*% W)
  return(y_hat)
}