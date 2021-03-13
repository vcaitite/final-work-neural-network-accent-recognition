rm(list=ls())
source("~/Documents/UFMG/9/Redes Neurais/TP2/final-work-neural-network-accent-recognition/ELM/trainELM.R")
source("~/Documents/UFMG/9/Redes Neurais/TP2/final-work-neural-network-accent-recognition/ELM/YELM.R")
source("~/Documents/UFMG/9/Redes Neurais/exemplos/escalonamento_matrix.R")
library(caret)

# Carregando base de dados:
path <- file.path("~/Documents/UFMG/9/Redes Neurais/TP2/final-work-neural-network-accent-recognition/databases", "treino.csv")
data_train <- read.csv(path)

executions <- 15

p <- c(228, 230, 232, 240, 243, 238) # número de neurônios

for (p in p){
  results <- rep(0, executions)
  for (index in 1:executions){
    # Separando dados de entrada e saída e treino e teste:
    particao <- createDataPartition(1:dim(data_train)[1],p=.7)
    train <- as.matrix(data_train[particao$Resample1,])
    validation <- as.matrix(data_train[- particao$Resample1,])
    
    x_train <- as.matrix(train[, 2:(ncol(train)-1)])
    y_train <- as.matrix(train[, ncol(train)])
    x_validation <- as.matrix(validation[, 2:(ncol(train)-1)])
    y_validation <- as.matrix(validation[, ncol(train)])
    
    # Escalonando os valores dos atributos para que fiquem restritos entre 0 e 1
    x_all <- rbind(x_train, x_validation)
    x_all <- staggeringMatrix(x_all, nrow(x_all), ncol(x_all))
    x_train <- x_all[1:nrow(x_train), ]
    x_validation <- x_all[(nrow(x_train)+1):(nrow(x_train)+nrow(x_validation)), ]
    
    
    length_train <- length(y_train)
    length_validation <- length(y_validation)
    
    # Treinando modelo:
    retlist<-trainELM(x_train, y_train, p, 1)
    W<-retlist[[1]]
    H<-retlist[[2]]
    Z<-retlist[[3]]
    
    # Calculando acurácia de treinamento
    y_hat_train <- as.matrix(YELM(x_train, Z, W, 1), nrow = length_train, ncol = 1)
    accuracy_train<-((sum(abs(y_hat_train + y_train)))/2)/length_train
    #print(paste("Acuracia de treinamento para p = ", p, " é ", accuracy_train))
      
    # Rodando dados de teste:
    y_hat_test <- as.matrix(YELM(x_validation, Z, W, 1), nrow = length_validation, ncol = 1)
    accuracy_validation<-((sum(abs(y_hat_test + y_validation)))/2)/length_validation
    results[index] <- accuracy_validation
    #
  }
  print(paste("Acuracia de teste para p = ", p, " é ", mean(results)))
}
  

