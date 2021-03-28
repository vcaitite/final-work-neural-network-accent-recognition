YMLP <- function(x_in, modMLP){
  W<-modMLP[[1]]
  Z<-modMLP[[2]] 
  u <- cbind(x_in,1) %*% Z
  H<-tanh(u)
  O<-cbind(H,1)%*%W
  yhat_test<-tanh(O)
  return(yhat_test)
}