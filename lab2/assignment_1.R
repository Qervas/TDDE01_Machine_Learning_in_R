## Question 1

data <- read.csv("tecator.csv", header = TRUE)
# Select only channels and Fat from original data
data_subset <- data[, 2:102]

set.seed(12345)

# Split data into training and test set
n <- dim(data_subset)[1]
id <- sample(1:n, floor(n*0.5))
train_data <- data_subset[id,]
test_data <- data_subset[-id,]


# Fit linear regression model on training data
lmFat = lm(Fat~., data = train_data)
summary(lmFat)

# Evaluate the model on training data
train_pred <- predict(lmFat, train_data)
train_MSE <- mean((train_pred - train_data$Fat)^2)
train_MSE

# Evaluate the model on test data
test_pred <- predict(lmFat, test_data)
test_MSE <- mean((test_pred - test_data$Fat)^2)
test_MSE

## Question 3

library(glmnet)
library(caret)

# Pick out features and response variables
features <- as.matrix(train_data[, 1:100])
response <- train_data$Fat

# Fit lasso regression model
lasso_m <- glmnet(features, response, alpha=1)
summary(lasso_m)
coef(lasso_m)


## Question 4

# Fit ridge regression model
ridge_m <- glmnet(features, response, alpha=0)
plot(ridge_m, xvar="lambda")

## Question 5

# Fit cross-validated lasso regression model
cv_lasso_m <- cv.glmnet(features, response, alpha=1)
plot(cv_lasso_m)

# Pick out optimal lambda
optimal_lambda <- cv_lasso_m$lambda.min
coef_optimal <- coef(cv_lasso_m, s=optimal_lambda)
num_variables <- sum(coef_optimal != 0)
cat("Num of variables used in model with optimal lambda: ", num_variables)

# Train lasso regression model with optimal lambda
optimal_m <- glmnet(features, response, alpha=1, lambda=optimal_lambda)
coef(optimal_m)
optimal_pred <- predict(optimal_m, s=optimal_lambda, newx=features)

plot(test_data$Fat, ylab="Predicted test values", 
     xlab="Original test values", col="blue")
points(optimal_pred, col="red")

