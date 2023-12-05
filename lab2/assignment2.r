setwd("C:/Masters/TDDE01-Machine Learning/Lab/Lab2")
library(tidyr)
library(tree)
library(rpart)
bank_data <- read.csv("bank-full.csv")
column_names <- colnames(bank_data) 
column_names <- unlist(strsplit(column_names,"[.]")) #fetching column names from bank data

new_bank_data <- separate(bank_data,1, column_names, sep =";") #data formatting

# removing column "balance" from data, as not part of input variable
new_bank_data <- new_bank_data[,-6] 


new_bank_data$age <- as.numeric(new_bank_data$age)
new_bank_data$job <- as.factor(new_bank_data$job)
new_bank_data$marital <- as.factor(new_bank_data$marital)
new_bank_data$education <- as.factor(new_bank_data$education)
new_bank_data$default <- as.factor(new_bank_data$default)
new_bank_data$housing <- as.factor(new_bank_data$housing)
new_bank_data$loan <- as.factor(new_bank_data$loan)
new_bank_data$contact <- as.factor(new_bank_data$contact)
new_bank_data$month <- as.factor(new_bank_data$month)
new_bank_data$day <-  as.factor(new_bank_data$day)
new_bank_data$duration <- as.numeric(new_bank_data$duration)
new_bank_data$campaign <- as.numeric(new_bank_data$campaign)
new_bank_data$pdays <- as.numeric(new_bank_data$pdays)
new_bank_data$previous <- as.numeric(new_bank_data$previous)
new_bank_data$poutcome <- as.factor(new_bank_data$poutcome)
new_bank_data$y <- ifelse(new_bank_data$y =="yes", 1,0)
new_bank_data$y <- as.factor(new_bank_data$y)

#remove input variable "duration"
processed_bank_data <- new_bank_data[, !names(new_bank_data) %in% c("duration")]

#####Task1#####
#dividing the data into training/validation/test
n <- dim(processed_bank_data)[1]
set.seed(12345)
id <- sample(1:n, floor(n*0.4))
training <- processed_bank_data[id,]

id1 <- setdiff(1:n, id)
set.seed(12345)
id2 <- sample(id1, floor(n*0.3))
validation <- processed_bank_data[id2,]

id3 <- setdiff(id1,id2)
test <- processed_bank_data[id3,]


####Task2#####

#Fit
#A: Decision Tree With Default Settings
default_tree <- tree(y~., data = training)

#Prediction with training and validation data(default tree).

prediction_deafult_train = predict(default_tree, training, type = "class")
prediction_deafult_valid = predict(default_tree, validation, type = "class")

#B: Decision Tree with smallest allowed node size equal to 7000
smallest_tree <- tree(y~., data = training, control = tree.control(nobs = nrow(training), minsize = 7000))

#Prediction with training and validation data(Smallest Allowed Node tree).

prediction_smallest_train = predict(smallest_tree, training, type = "class")
prediction_smallest_valid = predict(smallest_tree, validation, type = "class")

#C: Decision Trees minimum deviance to 0.0005.
mindev_tree <- tree(y~., data = training, control = tree.control(nobs = nrow(training), mindev = 0.0005))

#Prediction with training and validation data(Minimum deviance tree).

prediction_mindev_train = predict(mindev_tree, training, type = "class")
prediction_mindev_valid = predict(mindev_tree, validation, type = "class")

# function to calculate misclassifications rate
missclass_error <- function(X,Y){
  len <- length(X)
  return(1 - sum(diag(table(X,Y)))/len)
}

#Calculating misclassification on training data

miss_default <- missclass_error(training$y, prediction_deafult_train)
miss_small <- missclass_error(training$y, prediction_smallest_train)
miss_mindev <- missclass_error(training$y, prediction_mindev_train)

# calculate misclassification on validation data

miss_default_1 <- missclass_error(validation$y, prediction_deafult_valid)
miss_small_1 <- missclass_error(validation$y, prediction_smallest_valid)
miss_mindev_1 <- missclass_error(validation$y, prediction_mindev_valid)

missclass_rate <- data.frame("Default_Tree" = c(miss_default, miss_default_1),
                             "Smallest_Node" = c(miss_small,miss_small_1), 
                             "Minimum_Deviance" = c(miss_mindev,miss_mindev_1))
rownames(missclass_rate) <- c("Training","Validation")

missclass_rate


####Task3####

train_score <- rep(0,50)
valid_score <- rep(0,50)

for(i in 2:50)
{
  pruned_training <- prune.tree(mindev_tree, best = i)
  predict_pTree <- predict(pruned_training, validation, type = "tree")
  train_score[i] <- deviance(pruned_training)
  valid_score[i] <- deviance(predict_pTree)
}

plot(2:50, train_score[2:50], type = "b", col="red", ylim = c(0, 13000))
points(2:50, valid_score[2:50], type = "b", col="green")

#optimal amount of leaves

optimal_leaf <- which.min(valid_score[-1])

optimal_tree <- prune.tree(mindev_tree, best = optimal_leaf)
summary(optimal_tree)

plot(optimal_tree)


####Task4####

# test data prediction with optimal tree

prediction_test <- predict(optimal_tree, test, type = "class" )

#confusion matrix 
test_confusionMatrix <- table(true=test$y,prediction_test)
test_confusionMatrix

#Test Accuracy
test_cm_diag = sum(diag(test_confusionMatrix))
test_cm_sum = sum(test_confusionMatrix)

test_accuracy = test_cm_diag/test_cm_sum

test_accuracy

#F1 Score

test_recall <- test_confusionMatrix[2,2]/sum(test_confusionMatrix[2,])
test_precision <- test_confusionMatrix[2,2] / sum(test_confusionMatrix[,2])

F1_score <- 2*test_precision*test_recall / (test_precision+test_recall)
F1_score

####Task5####

loss_matrix <- matrix(c(0,5,1,0), nrow = 2)

lost_tree_model <- rpart(y~., data = training, method = "class", parms = list(loss= loss_matrix))

loss_predict_tree <- predict(lost_tree_model, test, type = "class")

#Confusion Matrix of Tree with Loss Matrix
loss_confusionMatrix <- table(true=test$y, predicted=loss_predict_tree)

loss_confusionMatrix

# Accuracy
loss_cm_diag = sum(diag(loss_confusionMatrix))
loss_cm_sum = sum(loss_confusionMatrix)

loss_accuracy = loss_cm_diag/loss_cm_sum
loss_accuracy

#F1 score

loss_recall <- loss_confusionMatrix[2,2]/sum(loss_confusionMatrix[2,])
loss_precision <- loss_confusionMatrix[2,2] / sum(loss_confusionMatrix[,2])

loss_F1_score <- 2*loss_precision*loss_recall / (loss_precision+loss_recall)
loss_F1_score

####Task6####

optimal_final_tree <- prune.tree(mindev_tree, best = 23)
optimal_tree_predection <- predict(optimal_final_tree, test, type = "vector")

#logistics regression
logistics <- glm(y~., data = training, family = "binomial")
logistics_prediction <- predict(logistics, test, type = "response")

#computing TPR and FPR with different thresholds

tree_TPR <- c()
tree_FPR <- c()
logistics_TPR <- c()
logistics_FPR <- c()

k <- 1

for(i in seq(0.05, 0.95, 0.05)){
  t_tree <- ifelse(optimal_tree_predection[,2]>i,1,0)
  t_logistics <- ifelse(logistics_prediction>i, 1,0)
  t1 <- table(test$y, t_tree)
  if(dim(t1)[2]>1){
    tree_TPR[k] <- t1[2,2] / sum(t1[2,])
    tree_FPR[k] <- t1[1,2] / sum(t1[1,])
  }
  
  t2 <- table(test$y, t_logistics)
  if(dim(t2)[2]>1){
    logistics_TPR[k] <- t2[2,2] / sum(t2[2,])
    logistics_FPR[k] <- t2[1,2] / sum(t2[1,])
  }
  k=k+1
}

#plotting ROC curves

plot(tree_FPR, tree_TPR, type = "l", col= "green",
     lwd=2,xlab = "False Positive rate",ylab = "True Positive rate", main = "ROC Curves")
lines(logistics_FPR, logistics_TPR, type = "l", col= "red", lwd=2)
legend("bottomright", c("Optimal Tree", "Logistics Regression"), fill= c("green","red"))
