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


