data <- read.csv("optdigits.csv", header = F)


set.seed(12345)
n <- dim(data)[1]

# Divide the data according to the partioning principle
train_ind <- sample(1:n, size = 0.5 * n)
valid_ind <- sample(setdiff(1:n, train_ind), size = 0.25 * n)
test_ind <- setdiff(1:n, c(train_ind, valid_ind))

# Change class labels to factor variable to indicate that is 
# a categorical variable and not a numerical variable
data[, 65] <- as.factor(data[, 65])

train_data <- data[train_ind,]
valid_data <- data[valid_ind,]
test_data <- data[test_ind,]

library(kknn)

# Misclassification function
misclass <- function(conf_matrix) {
  accuracy <- sum(diag(conf_matrix)) / sum(conf_matrix)
  error <- 1-accuracy
  return(error)
}

# Train KKNN classifier with k=30 and calculate misclassification
pred.kknn <- kknn(as.factor(V65)~., k=30, train=train_data, test=test_data, kernel="rectangular")
conf_matrix <- table(test_data[, 65], fitted.values(pred.kknn))
error <- misclass(conf_matrix)
print(round(error * 100, 2))
conf_matrix 

# Get prediction quality for digit 8
prob_label_eight <- pred.kknn$prob[,9]
easiest_preds <- order(prob_label_eight, decreasing=TRUE)[1:2]
hardest_preds <- order(prob_label_eight, decreasing=FALSE)[1:3]

# Visualize easiest predictions for digit 8
for (i in easiest_preds) {
  
  reshaped_matrix <- matrix(as.numeric(train_data[i, 1:64]),
                            nrow=8, ncol=8, byrow=T)
  heatmap(reshaped_matrix, Rowv=NA, Colv=NA)
  
}

# Visualize hardest predictions for digit 8
for (i in hardest_preds) {
  
  reshaped_matrix <- matrix(as.numeric(train_data[i, 1:64]),
                            nrow=8, ncol=8, byrow=T)
  heatmap(reshaped_matrix, Rowv=NA, Colv=NA)
  
}

train_errors <- numeric(30)
valid_errors <- numeric(30)

for (i in 1:30) {
  # Train KKNN on training data
  pred.kknn_tr <- kknn(as.factor(V65)~., train=train_data,
                       test=train_data, k=i, kernel="rectangular")
  conf_matrix_tr <- table(train_data[, 65], fitted.values(pred.kknn_tr))
  train_errors[i] <- misclass(conf_matrix_tr)
  
  # Train KKNN on validation data
  pred.kknn_va <- kknn(as.factor(V65)~., train=train_data,
                       test=valid_data, k=i, kernel="rectangular")
  conf_matrix_va <- table(valid_data[, 65], fitted.values(pred.kknn_va))
  valid_errors[i] <- misclass(conf_matrix_va)
}

# Plot the training and validation errors for different k's
plot(1:30, train_errors, col="blue", xlab="k",
     ylab="Misclass errors", main="Training and Validation errors")
points(1:30, valid_errors, col="red")

# Train KKNN with optimal k from plot and calculate misclassification error
pred.kknn_opt <- kknn(as.factor(V65)~., k=14, train=train_data,
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
optimal_k <- which.min(valid_errors)
print(paste("Optimal K: ", optimal_k))

