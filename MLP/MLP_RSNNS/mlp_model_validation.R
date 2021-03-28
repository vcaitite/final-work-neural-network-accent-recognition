rm(list=ls())
source("~/Documents/UFMG/9/Redes Neurais/exemplos/MLP_backpropagation_tanh_momentum.R")
source("~/Documents/UFMG/9/Redes Neurais/exemplos/YMLP_tanh.R")
library(caret)
library(RSNNS)

# Carregando base de dados:
path <- file.path("~/Documents/UFMG/9/Redes Neurais/TP2/final-work-neural-network-accent-recognition/databases", "treino.csv")
data_train <- read.csv(path)

executions <- 1

ps <- c(45) # número de neurônios

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
  x_all <- normalizeData(x_all, type = "norm")
  x_train <- x_all[1:nrow(x_train), ]
  x_validation <- x_all[(nrow(x_train)+1):(nrow(x_train)+nrow(x_validation)), ]
  
  
  length_train <- length(y_train)
  length_validation <- length(y_validation)
  for (p in ps){
    # Criando modelo:
    model <- mlp(x_train, y_train, size = p, maxit = 5000, initFunc = "Randomize_Weights",
                 initFuncParams = c(-0.3, 0.3), learnFunc = "Rprop",
                 learnFuncParams = c(0.1, 0.3, 0.1), updateFunc = "Topological_Order",
                 updateFuncParams = 0.1, hiddenActFunc = "Act_TanH",
                 shufflePatterns = TRUE, linOut = FALSE, inputsTest = NULL,
                 targetsTest = NULL)
    plotIterativeError(model)
    
    # Calculando acurácia de treinamento
    #y_hat_train <- predict(model, as.matrix(x_train))
    #yt <- (1*(y_hat_train >= 0.5)-0.5)*2
    #accuracy_train<-((sum(abs(yt + y_train)))/2)/length_train
    #print(paste("Acuracia de treinamento para p = ", p, " é ", accuracy_train))
    
    # Rodando dados de teste:
    y_hat_test <- predict(model, as.matrix(x_validation))
    plotRegressionError(y_hat_test, y_validation)
    yt <- (1*(y_hat_test >= 0.5)-0.5)*2
    accuracy_validation<-((sum(abs(yt + y_validation)))/2)/length_validation
    results[index, match(p, ps)] <- accuracy_validation
    print(paste("Acuracia de teste para p = ", p, " é ", accuracy_validation))
  }
}
print("-------------------------------------------------------------------------------------")

for (p in ps){
  print(paste("Acuracia de teste media para p = ", p, " é ", mean(results[, match(p, ps)])))
}
