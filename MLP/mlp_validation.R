rm(list=ls())
source("~/Documents/UFMG/9/Redes Neurais/exemplos/MLP_backpropagation_tanh_momentum.R")
source("~/Documents/UFMG/9/Redes Neurais/exemplos/YMLP_tanh.R")
source("~/Documents/UFMG/9/Redes Neurais/exemplos/escalonamento_matrix.R")
library(caret)

# Carregando base de dados:
path <- file.path("~/Documents/UFMG/9/Redes Neurais/TP2/final-work-neural-network-accent-recognition/databases", "treino.csv")
data_train <- read.csv(path)

executions <- 1

ps <- c(20) # número de neurônios

results <- matrix(rep(0, (executions*length(ps))), nrow = executions)
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
  for (p in ps){
    # Treinando modelo:
    modMLP<-backpropagation(x_train, y_train, p, 0.1, 1, 0.1, 2000)
    
    # Calculando acurácia de treinamento
    y_hat_train <- as.matrix(YMLP(x_train, modMLP), nrow = length_train, ncol = 1)
    yt <- (1*(y_hat_train >= 0)-0.5)*2
    accuracy_train<-((sum(abs(yt + y_train)))/2)/length_train
    #print(paste("Acuracia de treinamento para p = ", p, " é ", accuracy_train))
    
    # Rodando dados de teste:
    y_hat_test <- as.matrix(YMLP(x_validation, modMLP), nrow = length_test, ncol = 1)
    yt <- (1*(y_hat_test >= 0)-0.5)*2
    accuracy_validation<-((sum(abs(yt + y_validation)))/2)/length_validation
    results[index, match(p, ps)] <- accuracy_validation
    print(paste("Acuracia de teste para p = ", p, " é ", accuracy_validation))
  }
}
print("-------------------------------------------------------------------------------------")

for (p in ps){
  print(paste("Acuracia de teste media para p = ", p, " é ", mean(results[, match(p, ps)])))
}
