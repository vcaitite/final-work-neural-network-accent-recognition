backpropagation <- function(x_train, y_train, p, tol, eta, alfa, max_epoch){
  sech2<-function(u){
    return(((2/(exp(u)+exp(-u)))*(2/(exp(u)+exp(-u)))))
  }
  
  #Inicialização dos pesos.
  n <- ncol(x_train)
  N <- nrow(x_train)
  m <- ncol(y_train)
  
  #Matriz Z nxp, neste caso (n+1) x p 
  Z<-matrix(runif((n+1)*p)-0.5,ncol=p,nrow=n+1)
  Zt <- Z
  Ztless1 <- Z
  
  #Matriz W pxm, neste caso (p+1) x m 
  W<-matrix(runif((p+1)*1)-0.5,ncol=1,nrow=p+1)
  Wt <- W
  Wtless1 <- W
  
  x_actual <- matrix(nrow=(n+1),ncol=1)
  
  n_epoch <- 0
  error_epoch <- tol+1
  evec<-matrix(nrow=max_epoch,ncol=1)
  
  while((n_epoch < max_epoch) && (error_epoch > tol))
  {
    ei2<-0
    
    #Sequência aleatória de treinamento.
    xseq<-sample(N)
    for(i in 1:N)
    {
      #Amostra dado da sequência aleatória.
      irand <- xseq[i]
      x_actual[1:n,1] <- x_train[irand,]
      x_actual[n+1,1] <- 1
      
      y_actual <- y_train[irand, ]
      
      U<-t(x_actual)%*%Z
      
      H<-tanh(U)
      Haug<-cbind(H,1) 
      
      O<-Haug%*%W
      yhat <- tanh(O)
      
      error <- y_actual-yhat
      flinhaO <-sech2(O)
      dO <- error * flinhaO #Produto elemento a elemento
      
      Wminus <- W[-(p+1),]  #Saída do bias não se propaga
      ehidden <- dO%*%t(Wminus)
      flinhaU<-sech2(U)
      dU<-ehidden*flinhaU  #Produto elemento a elemento
      
      W<-Wt+eta*(t(Haug)%*%dO)+alfa*(Wt - Wtless1)
      Wtless1 <- Wt
      Wt <- W
      
      Z<-Zt+eta*(x_actual%*%dU)+alfa*(Zt - Ztless1)
      Ztless1 <- Zt
      Zt <- Z
      
      ei2<-ei2+(error%*%t(error))
    }
    #Incrementa número de épocas.
    n_epoch<-n_epoch+1
    evec[n_epoch]<-ei2/N
    #Armazena erro por época.
    error_epoch<-evec[n_epoch]
  }
  return(list(W, Z, evec, n_epoch))
}