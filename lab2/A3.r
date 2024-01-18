data = read.csv("communities.csv")
########task 1
library(ggplot2)
ViolentCrimesPerPop_col = data$ViolentCrimesPerPop
data_scaled = data[, names(data) != "ViolentCrimesPerPop"]
data_scaled = scale(data_scaled)
data_scaled = cbind(data_scaled, ViolentCrimesPerPop = ViolentCrimesPerPop_col)


PCA_fn = function(data_matrix){
  covariance_matrix = cov(data_matrix)
  eigen_results = eigen(covariance_matrix)
  eigenvalues = eigen_results$values
  eigenvectors = eigen_results$vectors
  explained_variance = eigenvalues / sum (eigenvalues)
  transformed_data = data_matrix %*% eigenvectors
  return (list(
      eigenvalues = eigenvalues,
      eigenvectors = eigenvectors,
      explained_variance = explained_variance,
      transformed_data = transformed_data
    ))
}

# Perform PCA
res = PCA_fn(data_scaled)

eigenvalues = res$eigenvalues
eigenvectors = res$eigenvectors
explained_variance = res$explained_variance
pca_data = res$transformed_data

#calculate cumulative explained variance
cum_explained_variance = cumsum(res$explained_variance)

#determine the number of components
num_components_95 = which(cum_explained_variance >= 0.95)[1]
paste("Proportiono of variance explained by first two components: ", res$explained_variance[1], ", ", res$explained_variance[2])
# "Proportion of variance explained by first two components:  0.250249400818273 ,  0.169308241196406"


##########Task 2###########
res = princomp(data_scaled, cor = TRUE)

#Trace Plot
plot(res$scores[, 1], type = 'p', main = 'Plot of the First Principal Component', xlab = 'Index', ylab = 'First Principal Component Score')

# get the loadings: the contribution of each feature to each principal component
loadings = res$loadings
abs_loadings_first_pc = abs(loadings[, 1])

#sort the loadings by ab in des order
sorted_loadings = sort(abs_loadings_first_pc, decreasing = TRUE)

top5_features = names(sorted_loadings)[1:5]
print("Top 5 features")
print(top5_features)
#"medFamInc"   "medIncome"   "PctKids2Par" "pctWInvInc"  "PctFam2Par" 
pc1_scores = res$scores[, "Comp.1"]
pc2_scores = res$scores[, "Comp.2"]

pca_score_df = data.frame(PC1 = pc1_scores, PC2 = pc2_scores, ViolentCrimes = ViolentCrimesPerPop_col)

ggplot(pca_score_df, aes(x = PC1, y = PC2, color = ViolentCrimesPerPop_col)) +
  geom_point() +
  scale_color_gradient(low = "blue", high = "red") + 
  labs(title = "PCA Scores Plot", x = "PC1", y = "PC2", color = "Vilent Crimes per Pop")+
  theme_minimal()


##########Task 3###########
library(caret)
library(caTools)
set.seed(123)

# Assuming 'data' is your dataset and 'ViolentCrimesPerPop' is the last column
# Scale the data except the response variable
features_scaled <- scale(data[, -ncol(data)])
response_scaled <- data[, ncol(data)]

# Split the data into training and test sets
index <- createDataPartition(response_scaled, p = 0.5, list = FALSE)
training_set <- data[index, ]
test_set <- data[-index, ]

# Fit a linear model with regularization (Lasso or Ridge) using the caret package
# Let's try Lasso here (L1 regularization)
model <- train(
  x = training_set[, -ncol(training_set)],
  y = training_set[, ncol(training_set)],
  method = "glmnet",
  trControl = trainControl("cv", number = 10), # 10-fold cross-validation
  tuneLength = 10 # Number of alpha/lambda combinations to try
)

# Evaluate the model
training_predictions <- predict(model, training_set[, -ncol(training_set)])
training_mse <- mean((training_set[, ncol(training_set)] - training_predictions)^2)
training_rmse <-sqrt(training_mse)

test_predictions <- predict(model, test_set[, -ncol(test_set)])
test_mse <- mean((test_set[, ncol(test_set)] - test_predictions)^2)
test_rmse <- sqrt(test_mse)
cat("Training MSE:", training_mse, "\n")
cat("Training RMSE:", training_rmse, "\n")
cat("Test MSE:", test_mse, "\n")
cat("Test RMSE:", test_rmse, "\n")
#Training MSE: 0.01523694 
#Training RMSE: 0.123438 
#Test MSE: 0.02009674 
#Test RMSE: 0.141763 

##############Task 4###############

training_features = as.matrix(training_set[, -ncol(training_set)])
test_features <- as.matrix(test_set[, -ncol(test_set)])
training_response <- training_set[, ncol(training_set)]
test_response <- test_set[, ncol(test_set)]

theta_start = rep(0, ncol(training_features))

# Global variables to store errors per iteration
training_errors <- c()
test_errors <- c()

# Modified cost function that also computes and stores errors
cost_function <- function(theta, features, response) {
  predictions <- features %*% theta
  mse <- mean((response - predictions)^2)
  
  # Store training and test errors at each iteration
  training_errors <<- c(training_errors, mse)
  test_predictions <- test_features %*% theta
  test_mse <- mean((test_response - test_predictions)^2)
  test_errors <<- c(test_errors, test_mse)
  
  return(mse)
}

# Run the optimization
optim_res <- optim(par = theta_start, fn = cost_function, 
                   features = training_features, response = training_response,
                   method = "BFGS", control = list(trace = 1, REPORT = 1, maxit = 1000))
#final  value 0.014527
# Plot training and test errors

plot(training_errors[-(1:500)], type = 'l', col = 'blue', xlab = 'Iteration', ylab = 'Error', main = 'Training and Test Errors per Iteration')
lines(test_errors[-(1:500)], col = 'red')
legend("topright", legend = c("Training Error", "Test Error"), col = c("blue", "red"), lty = 1)

# Identify the optimal iteration number
optimal_iter <- which.min(test_errors)
cat("Optimal Iteration Number:", optimal_iter, "\n")
#Optimal Iteration Number: 4953

# Plot smoothed training and test errors with a logarithmic scale
moving_average <- function(x, n = 5) {
  filter(x, rep(1/n, n), sides = 2)
}

# Apply moving average to training and test errors
smoothed_training_errors <- moving_average(training_errors, n = 50) # Adjust n as needed
smoothed_test_errors <- moving_average(test_errors, n = 50) # Adjust n as needed

plot(smoothed_training_errors[-(1:500)], type = 'l', col = 'blue', xlab = 'Iteration', ylab = 'Log(Error)', 
     main = 'Smoothed Training and Test Errors per Iteration (Log Scale)', log = "y")
lines(smoothed_test_errors[-(1:500)], col = 'red')
legend("topright", legend = c("Training Error", "Test Error"), col = c("blue","red"), lty = 1, y.intersp = 1.5)

valid_range <- (1 + 500):length(smoothed_training_errors)
plot(valid_range, smoothed_training_errors[valid_range], type = 'l', col = 'blue', xlab = 'Iteration', ylab = 'Log(Error)',
     main = 'Smoothed Training and Test Errors per Iteration (Log Scale)', log = "y", ylim = c(min(smoothed_training_errors[valid_range], smoothed_test_errors[valid_range], na.rm = TRUE), max(smoothed_training_errors, smoothed_test_errors, na.rm = TRUE)))
lines(valid_range, smoothed_test_errors[valid_range], col = 'red')
                                                                       
