rm(list=ls())
source("~/Documents/UFMG/9/Redes Neurais/TP2/final-work-neural-network-accent-recognition/RBF/trainRBF.R")
source("~/Documents/UFMG/9/Redes Neurais/TP2/final-work-neural-network-accent-recognition/RBF/YRBF.R")
source("~/Documents/UFMG/9/Redes Neurais/exemplos/escalonamento_matrix.R")
library(caret)

# Carregando base de dados:
path <- file.path("~/Documents/UFMG/9/Redes Neurais/TP2/final-work-neural-network-accent-recognition/databases", "treino.csv")
data_train <- read.csv(path)
path <- file.path("~/Documents/UFMG/9/Redes Neurais/TP2/final-work-neural-network-accent-recognition/databases", "teste.csv")
data_test <- read.csv(path)


# Separando dados de entrada e saída e treino e teste:
x_train <- as.matrix(data_train[1:3176, 2:20])
y_train <- as.matrix(data_train[1:3176, 21])
x_test <- as.matrix(data_test[1:1361, 2:20])

# Escalonando os valores dos atributos para que fiquem restritos entre 0 e 1
x_all <- rbind(x_train, x_test)
x_all <- staggeringMatrix(x_all, nrow(x_all), ncol(x_all))
x_train <- x_all[1:nrow(x_train), ]
x_test <- x_all[(nrow(x_train)+1):(nrow(x_train)+nrow(x_test)), ]

p <- 2100 # número de neurônios
executions <- 15
results <- matrix(nrow = nrow(x_test), ncol = executions)

for (index in 1:executions){
  # Treinando modelo:
  modRBF<-trainRBF(x_train, y_train, p)
  
  # Calculando acurácia de treinamento
  length_train <- length(y_train)
  y_hat_train <- as.matrix(YRBF(x_train, modRBF), nrow = length_train, ncol = 1)
  yt <- (1*(y_hat_train >= 0)-0.5)*2
  accuracy_train<-((sum(abs(yt + y_train)))/2)/length_train
  #print(accuracy_train)
  
  # Rodando dados de teste:
  y_hat_test <- as.matrix(YRBF(x_test, modRBF), nrow = length_test, ncol = 1)
  yt <- (1*(y_hat_test >= 0)-0.5)*2
  results[,index] <- yt
}

y <- rep(0, nrow(yt))
for (index in 1:nrow(yt)) {
  if(sum(results[index,] == 1) > (executions/2)){
    y[index] <- 1
  }
  else{
    y[index] <- -1
  }
}

Id <- 3177:4537
table <- data.frame(Id, y)
write.csv(table, "prediction_rbf.csv", row.names = FALSE)

