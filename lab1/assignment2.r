### Question 1
library(caret)
# Reading the data
parkinsons<-read.csv("parkinsons.csv")
new_parkinsons <- subset(parkinsons, select = -c(subject., age, sex, test_time, total_UPDRS))

# Dividing training and test data(60/40) 
n <- dim(new_parkinsons)[1]
set.seed(12345)
id <- sample(1:n, floor(n*0.6))
train <- new_parkinsons[id,]
test <- new_parkinsons[-id,]

#Scaling the data
scaler <- preProcess(train)
train_data <- predict(scaler, train)
test_data <- predict(scaler, test)

##Question 2

linear_modelp <- lm(motor_UPDRS~0 +., data = train_data)
summary(linear_modelp)

train_data_predict <- predict(linear_modelp, train_data)
train_MSE <- sum((train_data$motor_UPDRS - train_data_predict)^2) / nrow(train_data)

test_data_predict <- predict(linear_modelp, test_data)
test_MSE <- sum((test_data$motor_UPDRS - test_data_predict)^2) / nrow(test_data)

list(Traing_MSE = train_MSE, Test_MSE = test_MSE)

### Question 3

##(3.a) Loglikelihood

loglikelihood <- function(parameter)
{
  X <- as.matrix(train_data[,-1])
  n <- nrow(X)
  y <- train_data[,1]
  sigma <- parameter[17]
  theta <- parameter[1:16]
  return( -n/2 * log(2*pi) - n/2*log(sigma^2) - 1/(2*sigma^2) * sum((y - X%*%theta)^2))
}

##(3.b) Ridge Function

Ridge <- function(parameter, lambda)
{
  return( -loglikelihood(parameter) + lambda * sum(parameter^2))
}

##(3.c) RidgeOpt Function

RidgeOpt <- function(lambda)
{
  return(optim( rep(1,17), fn = Ridge, method = "BFGS", lambda = lambda))
}

##(3.d) Degree of Freedom

DF <- function(lambda)
{
  P <- as.matrix(train_data[,-1])
  degree <- P %*% solve(t(P) %*% P + lambda * diag(ncol(P))) %*% t(P)
  return(sum(diag(degree)))
}

## Question 4

train_P <- as.matrix(train_data[,-1])
test_P <- as.matrix(test_data[,-1])

# Prediction with lambda=1

ridgeopt_1 <- RidgeOpt(lambda = 1)

predict_train_1 <- train_P %*% ridgeopt_1$par[1:16]
predict_test_1 <- test_P %*% ridgeopt_1$par[1:16]

error_train_1 <- mean((predict_train_1 - train_data$motor_UPDRS)^2)
error_test_1 <- mean((predict_test_1 - test_data$motor_UPDRS)^2)

# Prediction with lambda=100

ridgeopt_100 <- RidgeOpt(lambda = 100)

predict_train_100 <- train_P %*% ridgeopt_100$par[1:16]
predict_test_100 <- test_P %*% ridgeopt_100$par[1:16]

error_train_100 <- mean((predict_train_100 - train_data$motor_UPDRS)^2)
error_test_100 <- mean((predict_test_100 - test_data$motor_UPDRS)^2)

# Prediction with lambda=100

ridgeopt_1000 <- RidgeOpt(lambda = 1000)

predict_train_1000 <- train_P %*% ridgeopt_1000$par[1:16]
predict_test_1000 <- test_P %*% ridgeopt_1000$par[1:16]

error_train_1000 <- mean((predict_train_1000 - train_data$motor_UPDRS)^2)
error_test_1000 <- mean((predict_test_1000 - test_data$motor_UPDRS)^2)

list(error_train_1 = error_train_1, error_train_100 = error_train_100, error_train_1000 = error_train_1000)

list(error_test_1 = error_test_1, error_test_100 = error_test_100, error_test_1000=error_test_1000)


degree_1 <- DF(1)
degree_100 <- DF(100)
degree_1000 <- DF(1000)

result <- data.frame( lambda = c(1,100,1000), 
                      MSE_train = c(error_train_1, error_train_100, error_train_1000), 
                      MSE_test = c(error_test_1,error_test_100,error_test_1000),
                      DF = c(degree_1, degree_100, degree_1000))

result
