##########Task 1###########
data = read.csv("communities.csv", header = TRUE)
data_scaled = scale(data)
# data_scaled[, names(data_scaled) != "ViolentCrimePerPop"] <- scale(data[, names(data) != "ViolentCrimesPerPop"])
performPCA <- function(data, exclude_column) {
    # Scale the variables except the specified column
    data_scaled <- data
    data_scaled[, names(data_scaled) != exclude_column] <- scale(data[, names(data) != exclude_column])

    # Compute the covariance matrix of the scaled data
    cov_matrix <- cov(data_scaled[, names(data_scaled) != exclude_column])

    # Compute eigenvalues and eigenvectors
    eigen_results <- eigen(cov_matrix)

    # Principal components
    principal_components <- eigen_results$vectors

    # Variance explained by each component
    variance_explained <- eigen_results$values / sum(eigen_results$values)

    # Determine the number of components for 95% variance
    cumulative_variance <- cumsum(variance_explained)
    components_for_95 <- which(cumulative_variance >= 0.95)[1]

    # Proportion of variation explained by the first two components
    first_two_components_variance <- variance_explained[1:2]

    # Results
    list(
        principal_components = principal_components,
        components_for_95_percent_variance = components_for_95,
        variance_explained_by_first_two = first_two_components_variance
    )
}

result <- performPCA(data, "ViolentCrimesPerPop")
result$components_for_95_percent_variance
result$variance_explained_by_first_two


##########Task 2###########
if(!require("MASS")) install.packages("MASS")
library(MASS)

pca_result <- princomp(data, cor = TRUE, scores = TRUE)
# Trace plot for the first principal component
plot(pca_result$scores[, 1], type = "l", main = "Trace Plot of the First Principal Component")

# Identifying the features contributing most to the first principal component
loadings_first_component <- abs(pca_result$loadings[, 1])
top_5_features <- sort(loadings_first_component, decreasing = TRUE)[1:5]

# Print the top 5 features
print(top_5_features)


# Install and load necessary packages
if (!require("ggplot2")) install.packages("ggplot2")
library(ggplot2)

# Perform PCA using princomp() - make sure to exclude the ViolentCrimesPerPop variable from scaling
pca_result <- princomp(data[-which(names(data) == "ViolentCrimesPerPop")], cor = TRUE, scores = TRUE)

# Create a data frame for ggplot
plot_data <- data.frame(PC1 = pca_result$scores[, 1],
                        PC2 = pca_result$scores[, 2],
                        ViolentCrimesPerPop = data$ViolentCrimesPerPop)

# Plot using ggplot2
ggplot(plot_data, aes(x = PC1, y = PC2, color = ViolentCrimesPerPop)) +
    geom_point() +
    labs(title = "PCA Plot with Violent Crimes Per Population",
         x = "Principal Component 1",
         y = "Principal Component 2") +
    scale_color_gradient(low = "blue", high = "red") # color gradient from low to high values


##########Task 3###########
# Load necessary libraries
if (!require("caret")) install.packages("caret")
library(caret)

# Assuming your data is stored in a data frame called 'data'
set.seed(12345) # For reproducibility
indexes <- createDataPartition(data$ViolentCrimesPerPop, p = 0.5, list = FALSE)
train_data <- data[indexes, ]
test_data <- data[-indexes, ]

# Scale the data
preproc <- preProcess(train_data[, -which(names(train_data) == "ViolentCrimesPerPop")], method = c("center", "scale"))
train_data_scaled <- predict(preproc, train_data)
test_data_scaled <- predict(preproc, test_data)

# Scaling the response variable
train_response_mean <- mean(train_data$ViolentCrimesPerPop)
train_response_sd <- sd(train_data$ViolentCrimesPerPop)
train_data_scaled$ViolentCrimesPerPop <- (train_data$ViolentCrimesPerPop - train_response_mean) / train_response_sd
test_data_scaled$ViolentCrimesPerPop <- (test_data$ViolentCrimesPerPop - train_response_mean) / train_response_sd

# Fit linear regression model
lm_model <- lm(ViolentCrimesPerPop ~ ., data = train_data_scaled)

# Make predictions
train_predictions <- predict(lm_model, train_data_scaled)
test_predictions <- predict(lm_model, test_data_scaled)

# Compute MSE
train_mse <- mean((train_data_scaled$ViolentCrimesPerPop - train_predictions)^2)
test_mse <- mean((test_data_scaled$ViolentCrimesPerPop - test_predictions)^2)

# Output the errors
list(train_mse = train_mse, test_mse = test_mse)

plot_train_data <- data.frame(Actual = train_data_scaled$ViolentCrimesPerPop, Predicted = train_predictions)
plot_test_data <- data.frame(Actual = test_data_scaled$ViolentCrimesPerPop, Predicted = test_predictions)

# Plot for training data
p1 <- ggplot(plot_train_data, aes(x = Actual, y = Predicted)) +
    geom_point(color = 'blue', alpha = 0.5) +
    geom_abline(intercept = 0, slope = 1, color = 'red', linetype = "dashed") +
    labs(title = "Training Data: Actual vs Predicted",
         x = "Actual values",
         y = "Predicted values") +
    theme_minimal()

# Plot for test data
p2 <- ggplot(plot_test_data, aes(x = Actual, y = Predicted)) +
    geom_point(color = 'green', alpha = 0.5) +
    geom_abline(intercept = 0, slope = 1, color = 'red', linetype = "dashed") +
    labs(title = "Test Data: Actual vs Predicted",
         x = "Actual values",
         y = "Predicted values") +
    theme_minimal()

# Arrange the plots into one figure
library(gridExtra)
grid.arrange(p1, p2, ncol = 2)



##############Task 4###############

# Global variables to store the training and test errors
train_errors <<- numeric(0)
test_errors <<- numeric(0)
theta_history <<- list()
# Define the cost function for linear regression without intercept
cost_function <- function(theta, X, y) {
  predictions <- X %*% theta
  mse <- mean((predictions - y) ^ 2)
  return(mse)
}

# Wrapper function to update errors at each iteration
cost_function_with_errors <- function(theta) {

  theta_history <<- c(theta_history, list(theta))
  # Calculate the training MSE using the provided parameters
  train_mse <- cost_function(theta, X_train, y_train)
  
  # Append the training error to the global train_errors variable
  train_errors <<- c(train_errors, train_mse)
  
  # Calculate the test MSE using the provided parameters
  test_predictions <- X_test %*% theta
  test_mse <- mean((test_predictions - y_test) ^ 2)
  
  # Append the test error to the global test_errors variable
  test_errors <<- c(test_errors, test_mse)
  
  return(train_mse)
}

# Prepare the data
X_train <- as.matrix(train_data_scaled[, -which(names(train_data_scaled) == "ViolentCrimesPerPop")])
y_train <- train_data_scaled$ViolentCrimesPerPop
X_test <- as.matrix(test_data_scaled[, -which(names(test_data_scaled) == "ViolentCrimesPerPop")])
y_test <- test_data_scaled$ViolentCrimesPerPop

# Run the optimization using the wrapper function
optim_results <- optim(
  par = rep(0, ncol(X_train)), # Initial parameters set to zero
  fn = cost_function_with_errors, # Use the wrapper function
  method = "BFGS",
  control = list(maxit = 1000, REPORT = 1) # Report progress at every iteration
)

# Create a data frame for plotting errors
iterations <- 1:length(train_errors)  # Number of iterations tracked
plot_data <- data.frame(
  Iteration = iterations,
  TrainError = train_errors,
  TestError = test_errors
)

# Skip the first few iterations if they have extreme values
plot_data_filtered <- plot_data[-(1:10), ]  

# Apply a logarithmic transformation to the errors
plot_data_filtered$TrainError <- log(plot_data_filtered$TrainError + 1)  # Adding 1 to avoid log(0)
plot_data_filtered$TestError <- log(plot_data_filtered$TestError + 1)

# Generate the plot with a logarithmic scale and smoothed lines
ggplot(plot_data_filtered, aes(x = Iteration)) +
  geom_line(aes(y = TrainError, color = "Training Error"), stat = "smooth") +
  geom_line(aes(y = TestError, color = "Test Error"), stat = "smooth") +
  scale_y_log10(labels = scales::comma) +  # Logarithmic scale
  labs(title = "Training and Test Error over Iterations (Log Scale)",
       x = "Iteration Number",
       y = "Log Error") +
  scale_color_manual("", 
                     breaks = c("Training Error", "Test Error"),
                     values = c("Training Error" = "blue", "Test Error" = "red")) +
  theme_minimal()
  
find_early_stopping_point <- function(test_errors) {
  stop_point <- which.min(test_errors)
  return(stop_point)
}
# Run the optimization using the wrapper function
optim_results <- optim(
  par = rep(0, ncol(X_train)),
  fn = cost_function_with_errors,
  method = "BFGS",
  control = list(maxit = 1000, REPORT = 1)
)

# Find the early stopping iteration
early_stop_iteration <- find_early_stopping_point(test_errors)

# Use the optimal model parameters
optimal_theta <- theta_history[[early_stop_iteration]]

# Compute the train and test losses with the optimal model
optimal_train_loss <- cost_function(optimal_theta, X_train, y_train)
optimal_test_loss <- cost_function(optimal_theta, X_test, y_test)

# Print the results
cat("Early Stopping at Iteration:", early_stop_iteration, "\n")
cat("Optimal Train Loss:", optimal_train_loss, "\n")
cat("Optimal Test Loss:", optimal_test_loss, "\n")
