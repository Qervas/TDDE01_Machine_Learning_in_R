setwd("C:/Masters/TDDE01-Machine Learning/Lab/Lab3")

library(neuralnet)
library(ggplot2)
set.seed(1234567890)

####Task1####

Var <- runif(500, 0 , 10)
mydata <- data.frame(Var, Sin= sin(Var))

tr <- mydata[1:25,] #Training Data
te <- mydata[26:500,] #Test Data

tr_nn <- neuralnet(formula = Sin ~Var, data = tr, hidden = 10)

predict_te <- predict(tr_nn, te)

plot(tr, col= "blue",cex = 1.5)
points(te, col = "green")
points(te[,1],predict_te, col="orange")
legend("bottomleft", c("Training", "Test", "Predicted"), fill= c("blue","green", "orange"))



####Task2####

linear_activation <- function(x)
{
  y <- x
}

linear_tr_nn <- neuralnet(formula = Sin ~ Var, data = tr, hidden = 10, act.fct = linear_activation)
predict_tr_h1 <- predict(linear_tr_nn, te)

plot(tr, col = "blue", cex= 2)
points(te, col = "green", cex= 1)
points(te[,1], predict_tr_h1, col = "orange" , cex= 1 )

relu_activation <- function(x)
{
  ifelse(x>0, x,0)
}

relu_nn <- neuralnet(formula = Sin ~ Var, data = tr, hidden = 10, act.fct = relu_activation)
predict_te_h2 <- predict(relu_nn, te)

plot(tr, col = "blue", cex= 2)
points(te, col = "green", cex= 1)
points(te[,1], predict_te_h2, col = "orange" , cex= 1 )

softmax_activation <- function(x)
{
  y = log(1 + exp(x))
}

softmax_nn <- neuralnet(formula = Sin ~ Var, data = tr, hidden = 10, act.fct = softmax_activation)
predict_te_h3 <- predict(softmax_nn, te)

plot(tr, col = "blue", cex= 2)
points(te, col = "green", cex= 1)
points(te[,1], predict_te_h3, col = "orange" , cex= 1 )


####Task3####

Var <- runif(500, 0, 50)
mydata_1 <- data.frame(Var, Sin = sin(Var))

predict_tr_1 <- predict(tr_nn, mydata_1 )

plot(mydata_1, col="blue", ylim= c(-4,4))
points(mydata_1[,1], predict_tr_1, col="green")

####Task4####

tr_nn$weights

####Task5####

Var_1 <- runif(500, 0, 10)
mydata_2 <- data.frame(Var_1, Sin = sin(Var_1))
tr_nn_new <- neuralnet(formula = Var_1~Sin, data = mydata_2, threshold = 0.1) 

predict_new <- predict(tr_nn_new, mydata_2)

plot(mydata_2, col = "blue", cex = 2, ylim = c(-5,10))
points(mydata_2[,1], predict_new, col = "green", cex=2)
