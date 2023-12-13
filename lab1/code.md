# Assignment 1

```R
data <- read.csv("optdigits.csv", header=FALSE)
data$V65 <- as.factor(data$V65)


# Divide the data according to the partioning principle
n <- dim(data)[1]
set.seed(12345)
id <- sample(1:n, floor(n*0.5))
train_data <- data[id, ]

id1 <- setdiff(1:n, id)
set.seed(12345)
id2 <- sample(id1, floor(n*0.25))
valid_data <- data[id2, ]

id3 <- setdiff(id1, id2)
test_data <- data[id3, ] 

library(kknn)

# Misclassification function
misclass <- function(conf_matrix) {
  accuracy <- sum(diag(conf_matrix)) / sum(conf_matrix)
  error <- 1-accuracy
  return(error)
}

# Train KKNN classifier with k=30 and calculate misclassification
pred.kknn_test <- kknn(as.factor(V65)~., k=30, train=train_data, test=test_data, kernel="rectangular")
pred.kknn_train <- kknn(as.factor(V65)~., k=30, train=train_data, test=train_data, kernel="rectangular")

conf_matrix_test <- table(test_data$V65, fitted.values(pred.kknn_test))
conf_matrix_train <- table(train_data$V65, fitted.values(pred.kknn_train))

error_test <- misclass(conf_matrix_test)
error_train <- misclass(conf_matrix_train)

print(round(error_train * 100, 2))
print(round(error_test * 100, 2))

# Get prediction quality for digit 8
train_eights <- subset(train_data, V65 == "8")
prob_eights <- subset(pred.kknn_train[["prob"]], train_data$V65 == "8")

hardest_eights <- order(prob_eights[, "8"])[1:3]
easiest_eights <- order(prob_eights[, "8"], decreasing=TRUE)[1:2]
combined <- c(hardest_eights, easiest_eights)

for (i in 1:5) {
  
  reshaped_matrix <- matrix(as.numeric(train_eights[combined[i], -65]),
                            nrow=8, ncol=8, byrow=T)
  heatmap(reshaped_matrix, Rowv=NA, Colv=NA)
}

train_errors <- numeric(30)
valid_errors <- numeric(30)

for (i in 1:30) {
  # Train KKNN on training and valid data
  pred.kknn_tr <- kknn(as.factor(V65)~., train=train_data,
                       test=train_data, k=i, kernel="rectangular")
  pred.kknn_va <- kknn(as.factor(V65)~., train=train_data,
                       test=valid_data, k=i, kernel="rectangular")
  
  conf_matrix_tr <- table(train_data[, 65], fitted.values(pred.kknn_tr))
  conf_matrix_va <- table(valid_data[, 65], fitted.values(pred.kknn_va))
  
  valid_errors[i] <- misclass(conf_matrix_va)
  train_errors[i] <- misclass(conf_matrix_tr)
  
}

# Plot the training and validation errors for different k's
plot(train_errors*100, col="blue", xlab="k",
     ylab="Misclass errors", main="Training and Validation errors")
points(valid_errors*100, col="red")

# Train KKNN with optimal k from plot and calculate misclassification error

opt_k <- which.min(valid_errors)
print(opt_k)
pred.kknn_opt <- kknn(as.factor(V65)~., k=opt_k, train=train_data,
                      test=test_data, kernel="rectangular")
conf_matrix_opt <- table(test_data[, 65], fitted.values(pred.kknn_opt))
test_error <- misclass(conf_matrix_opt)
print(round(test_error * 100, 2))


valid_errors <- numeric(30)

for (i in 1:30) {

  pred.kknn_va <- kknn(as.factor(V65)~., train=train_data,
                       test=valid_data, k=i, kernel="rectangular")
  
  # Cross-entropy calculation
  for (digit in 0:9) {
  
    valid_probs <- pred.kknn_va$prob[which(valid_data$V65 == digit), digit+1]
    valid_probs <- sum(sapply(valid_probs, function(x) -log(x + 1e-15)))
    valid_errors[i] <- valid_errors[i] + valid_probs
  }

}

plot(1:30, valid_errors, col="blue", xlab="k", type="b",
     ylab="Cross-entropy errors", main="Cross-entropy errors for different K")



```

---

## Assignment 2

```r
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

```

---

## Assignment 3

```r
library(ggplot2)
library(caret)

# Define column labels for the data frame
column_labels <- c(
  'Number_of_times_pregnant',
  'Plasma_glucose_concentration',
  'Diastolic_blood_pressure',
  'Triceps_skinfold_thickness',
  'Two_Hour_serum_insulin',
  'Body_mass_index',
  'Diabetes_pedigree_function',
  'Age',
  'Diabetes'
)

# Read the CSV file without headers
df <- read.csv('pima-indians-diabetes.csv', header = FALSE, col.names = column_labels)

##################################Task 1##################################
# Scatterplot with ggplot2
p <- ggplot(df, aes(x = Age, y = Plasma_glucose_concentration, color = factor(Diabetes))) +
  geom_point(alpha = 0.5) +
  scale_color_manual(values = c('0' = 'blue', '1' = 'red')) +
  labs(title = 'Plasma Glucose Concentration on Age Colored by Diabetes Level',
       x = 'Age',
       y = 'Plasma Glucose Concentration',
       color = 'Diabetes Level') +
  theme_minimal() +
  theme(legend.position = "bottom")

print(p)


##################################Task 2##################################
library(dplyr)
library(ggplot2)
library(caret)

df$Diabetes <- as.factor(df$Diabetes)

# Outlier removal based on the IQR
calculate_bounds <- function(x) {
  Q <- quantile(x, probs = c(.25, .75), na.rm = TRUE)
  iqr <- IQR(x, na.rm = TRUE)
  return(c(Q[1] - 1.5 * iqr, Q[2] + 1.5 * iqr))
}

# Apply the function to Plasma_glucose_concentration and Age
glucose_bounds <- calculate_bounds(df$Plasma_glucose_concentration)
age_bounds <- calculate_bounds(df$Age)

# Filter out the outliers
df_filtered <- df %>%
  filter(Plasma_glucose_concentration >= glucose_bounds[1] & Plasma_glucose_concentration <= glucose_bounds[2]) %>%
  filter(Age >= age_bounds[1] & Age <= age_bounds[2])

# Feature scaling using Z-score standardization
df_filtered <- df_filtered %>%
  mutate(
    Scaled_Glucose = scale(Plasma_glucose_concentration),
    Scaled_Age = scale(Age)
  )

# Prepare the data for logistic regression
df_logistic <- df_filtered[, c('Scaled_Glucose', 'Scaled_Age', 'Diabetes')]

# Split the data into training and test sets
set.seed(42)
index <- createDataPartition(df_logistic$Diabetes, p = 0.8, list = FALSE)
train_data <- df_logistic[index, ]
test_data <- df_logistic[-index, ]

# Fit the logistic regression model
model <- glm(Diabetes ~ Scaled_Age + Scaled_Glucose, data = train_data, family = "binomial")

# Summary to check for convergence
summary(model)

# Predict probabilities
test_data$prob <- predict(model, newdata = test_data, type = "response")

# Apply the classification threshold
test_data$Diabetes_pred <- ifelse(test_data$prob >= 0.5, '1', '0')

# Convert predictions to a factor for consistency with the actual values
test_data$Diabetes_pred <- factor(test_data$Diabetes_pred, levels = levels(df$Diabetes))

# Calculate misclassification error
misclassification_error <- mean(test_data$Diabetes != test_data$Diabetes_pred)
cat(sprintf("Misclassification error: %f\n", misclassification_error))

# Plotting with scaled features
ggplot(test_data, aes(x = Scaled_Age, y = Scaled_Glucose, color = Diabetes_pred)) +
  geom_point(alpha = 0.5) +
  scale_color_manual(values = c('1' = 'red', '0' = 'blue')) +
  labs(title = 'Scaled Age vs Scaled Plasma Glucose Concentration Colored by Predicted Diabetes Status',
       x = 'Scaled Age',
       y = 'Scaled Plasma Glucose Concentration',
       color = 'Diabetes Status') +
  theme_minimal() +
  theme(legend.position = "bottom")


##################################Task 3##################################
library(ggplot2)
library(dplyr)
library(reshape2)

# Create a meshgrid for the contour plot
age_range <- range(df$Age)
glucose_range <- range(df$Plasma_glucose_concentration) 

age_seq <- seq(from = age_range[1] - 1, to = age_range[2] + 1, by = 0.1)
glucose_seq <- seq(from = glucose_range[1] - 1, to = glucose_range[2] + 1, by = 0.1)

grid <- expand.grid(Age = age_seq, Plasma_glucose_concentration = glucose_seq)

# Predict probabilities on the meshgrid
grid$prob <- predict(model, newdata = grid, type = "response")

# Reshape for ggplot
grid_melt <- melt(grid, id.vars = c("Age", "Plasma_glucose_concentration"))

# Plotting
ggplot() +
  geom_tile(data = grid_melt, aes(x = Age, y = Plasma_glucose_concentration, fill = value), alpha = 0.5) +
  scale_fill_gradient2(low = "blue", high = "red", mid = "white", midpoint = 0.5, limit = c(0, 1), space = "Lab", name="Probability") +
  geom_contour(data = grid_melt, aes(x = Age, y = Plasma_glucose_concentration, z = value), breaks = c(0.5), color = "grey") +
  geom_point(data = test_data, aes(x = Age, y = Plasma_glucose_concentration, color = as.factor(Diabetes_pred)), alpha = 0.5) +
  labs(title = 'Age vs Plasma Glucose Concentration with Decision Boundary',
       x = 'Age',
       y = 'Plasma Glucose Concentration') +
  theme_minimal() +
  theme(legend.position = "bottom")


##################################Task 4##################################
library(ggplot2)
library(dplyr)
library(caret)

# Assuming that df has already had outliers removed and features scaled as per the previous step
df$Diabetes <- as.factor(df_filtered$Diabetes)

# Prepare the data for logistic regression
df_logistic <- df_filtered[, c('Scaled_Glucose', 'Scaled_Age', 'Diabetes')]

# Split the data into training and test sets
set.seed(42)
index <- createDataPartition(df_logistic$Diabetes, p = 0.8, list = FALSE)
train_data <- df_logistic[index, ]
test_data <- df_logistic[-index, ]

# Fit the logistic regression model to the preprocessed data
model <- glm(Diabetes ~ Scaled_Age + Scaled_Glucose, data = train_data, family = "binomial")

# Predict probabilities on the test data (scaled)
test_data$prob <- predict(model, newdata = test_data, type = "response")

# Apply the lower threshold
test_data$y_pred_low_threshold <- ifelse(test_data$prob >= 0.2, '1', '0')
test_data$y_pred_low_threshold <- factor(test_data$y_pred_low_threshold, levels = levels(df$Diabetes))

# Apply the higher threshold
test_data$y_pred_high_threshold <- ifelse(test_data$prob >= 0.8, '1', '0')
test_data$y_pred_high_threshold <- factor(test_data$y_pred_high_threshold, levels = levels(df$Diabetes))

# Plot for lower threshold
p1 <- ggplot(test_data, aes(x = Scaled_Age, y = Scaled_Glucose, color = y_pred_low_threshold)) +
  geom_point(alpha = 0.5) +
  scale_color_manual(values = c('1' = 'red', '0' = 'blue')) +
  labs(title = 'Scaled Age vs Scaled Plasma Glucose Concentration with r = 0.2',
       x = 'Scaled Age',
       y = 'Scaled Plasma Glucose Concentration',
       color = 'Predicted Class with r = 0.2') +
  theme_minimal() +
  theme(legend.position = "bottom")

# Plot for higher threshold
p2 <- ggplot(test_data, aes(x = Scaled_Age, y = Scaled_Glucose, color = y_pred_high_threshold)) +
  geom_point(alpha = 0.5) +
  scale_color_manual(values = c('1' = 'red', '0' = 'blue')) +
  labs(title = 'Scaled Age vs Scaled Plasma Glucose Concentration with r = 0.8',
       x = 'Scaled Age',
       y = 'Scaled Plasma Glucose Concentration',
       color = 'Predicted Class with r = 0.8') +
  theme_minimal() +
  theme(legend.position = "bottom")

# Print the plots
print(p1)
print(p2)


##################################Task 5##################################
library(ggplot2)
library(caret)
library(dplyr)

# Assuming 'df_filtered' is your DataFrame and it already includes 'x1' and 'x2'
df_filtered$z1 <- df_filtered$Plasma_glucose_concentration^4
df_filtered$z2 <- df_filtered$Plasma_glucose_concentration^3 * df_filtered$Age
df_filtered$z3 <- df_filtered$Plasma_glucose_concentration^2 * df_filtered$Age^2
df_filtered$z4 <- df_filtered$Plasma_glucose_concentration * df_filtered$Age^3
df_filtered$z5 <- df_filtered$Age^4

# Define the features and the target variable
X <- df_filtered[, c('Plasma_glucose_concentration', 'Age', 'z1', 'z2', 'z3', 'z4', 'z5')]
y <- df_filtered$Diabetes

# Split the data into training and test sets
set.seed(42)  # For reproducibility
train_index <- createDataPartition(y, p = 0.8, list = FALSE)
X_train <- X[train_index, ]
y_train <- y[train_index]
X_test <- X[-train_index, ]
y_test <- y[-train_index]

# Train the model
model <- glm(y_train ~ ., data = as.data.frame(X_train), family = 'binomial')

# Predict on the training set
y_train_pred <- predict(model, newdata = X_train, type = "response")
y_train_pred_class <- ifelse(y_train_pred > 0.5, 1, 0)

# Compute the misclassification rate
misclassification_rate <- mean(y_train != y_train_pred_class)
cat(sprintf("Misclassification Rate on Training Set: %f\n", misclassification_rate))

# Plotting
# Convert predictions to a factor to match the actual y values
y_train_pred_class <- factor(ifelse(y_train_pred > 0.5, 1, 0), levels = c(0, 1))

# Include the predicted classes into the training data frame for plotting
X_train$PredictedClass <- y_train_pred_class

# Plotting with Age on the x-axis and Plasma Glucose Concentration on the y-axis
ggplot(X_train, aes(x = Age, y = Plasma_glucose_concentration, color = PredictedClass)) +
  geom_point(alpha = 0.5) +
  scale_color_manual(values = c('0' = 'blue', '1' = 'red')) +
  labs(title = 'Age vs Plasma Glucose Concentration with Polynomial Features',
       x = 'Age',
       y = 'Plasma Glucose Concentration',
       color = 'Predicted Class') +
  theme_minimal() +
  theme(legend.position = "bottom")




```
