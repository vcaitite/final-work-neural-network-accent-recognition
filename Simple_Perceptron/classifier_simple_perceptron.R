rm(list=ls())
source("~/Documents/UFMG/9/Redes Neurais/TP2/final-work-neural-network-accent-recognition/Simple_Perceptron/trainPerceptron.R")
source("~/Documents/UFMG/9/Redes Neurais/TP2/final-work-neural-network-accent-recognition/Simple_Perceptron/yperceptron.R")
source("~/Documents/UFMG/9/Redes Neurais/exemplos/escalonamento_matrix.R")
library(caret)

# Carregando base de dados:
path <- file.path("~/Documents/UFMG/9/Redes Neurais/TP2/final-work-neural-network-accent-recognition/databases", "treino.csv")
data_train <- read.csv(path)
path <- file.path("~/Documents/UFMG/9/Redes Neurais/TP2/final-work-neural-network-accent-recognition/databases", "teste.csv")
data_test <- read.csv(path)

# Separando dados de entrada e saída e treino e teste:
x_train <- as.matrix(data_train[1:3176, 2:20])
class <- as.matrix(data_train[1:3176, 21])
y_train <- rep(0,nrow(x_train))
for (count in 1:length(class)) {
  if (class[count] == 1 ){
    y_train[count] <- 1
  } 
  else{
    y_train[count] <- 0
  }
}
x_test <- as.matrix(data_test[1:1361, 2:20])

# Escalonando os valores dos atributos para que fiquem restritos entre 0 e 1
x_all <- rbind(x_train, x_test)
x_all <- staggeringMatrix(x_all, nrow(x_all), ncol(x_all))
x_train <- x_all[1:nrow(x_train), ]
x_test <- x_all[(nrow(x_train)+1):(nrow(x_train)+nrow(x_test)), ]

# Treinando modelo:
retlist<-trainPerceptron(x_train, y_train, 0.2, 0.1, 10000, 1)
W<-retlist[[1]]

# Calculando acurácia de treinamento
length_train <- length(y_train)
y_hat_train <- as.matrix(yperceptron(x_train, W, 1), nrow = length_train, ncol = 1)
accuracy_train <- 1-((t(y_hat_train-y_train) %*% (y_hat_train-y_train))/length_train)

# Rodando dados de teste:
y_hat_test <- as.matrix(yperceptron(x_test, W, 1), nrow = length_test, ncol = 1)
y <- ifelse(y_hat_test == 0, -1, 1) 

Id <- 3177:4537
table <- data.frame(Id, y)
write.csv(table, "prediction_perceptron.csv", row.names = FALSE)

