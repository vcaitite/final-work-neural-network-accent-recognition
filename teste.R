rm(list=ls())
library(neuralnet)
source("~/Documents/UFMG/9/Redes Neurais/exemplos/escalonamento_matrix.R")

# Carregando base de dados:
path <- file.path("~/Documents/UFMG/9/Redes Neurais/TP2/final-work-neural-network-accent-recognition/databases", "treino.csv")
data_train <- read.csv(path)
path <- file.path("~/Documents/UFMG/9/Redes Neurais/TP2/final-work-neural-network-accent-recognition/databases", "teste.csv")
data_test <- read.csv(path)

data_train <- data_train[-c(1)]

particao = createDataPartition(1:dim(data_train)[1],p=.7)
x_train_t = data_train[particao$Resample1,]
x_train_v = data_train[- particao$Resample1,]

formula <- y ~ X0 + X1 + X2 + X3 + X4 + X5 + X6 + X7 + X8 + X9 + X10 + X11 + X12 + X13 + X14 + X15 + X16 + X17 + X18
formula_fselection <- y ~ X4 + X3 + X6 + X2 + X13 + X5 + X15 + X14 + X16 + X12
modelo = neuralnet(formula = formula, x_train_t , hidden=c(150), stepmax=1e7)

print(modelo)
plot(modelo)

teste = compute(modelo, x_train_v[,1:19])

result = teste$net.result

resultado = as.data.frame(teste$net.result)

resultado$class = colnames(resultado[,1])[max.col(resultado[,1], ties.method = 'first')]

for (index in 1:nrow(result)) {
  if(result[index] >= 0){
    result[index] <- 1
  }
  else{
    result[index] <- -1
  }
}
accuracy_train<-((sum(abs(result+x_train_v$y)))/2)/nrow(x_train_v)



