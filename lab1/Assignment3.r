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
library(ggplot2)
library(caret)
library(dplyr)

# Assuming df is your DataFrame with the correct columns
x1 <- df$Plasma_glucose_concentration
x2 <- df$Age
r <- 0.5  # Classification threshold
y <- df$Diabetes

# Prepare the data for logistic regression
df_logistic <- data.frame(x1 = x1, x2 = x2, y = as.factor(y))

# Split the data into training and test sets
set.seed(42)  # For reproducibility
index <- createDataPartition(df_logistic$y, p = 0.8, list = FALSE)
train_data <- df_logistic[index, ]
test_data <- df_logistic[-index, ]

# Fit the logistic regression model
model <- glm(y ~ x1 + x2, data = train_data, family = "binomial")

# Output the number of iterations (may not be directly available like in Python)
cat("Number of iterations: Not directly available in R's glm function\n")

# Coefficients
b0 <- coef(model)[1]
b1 <- coef(model)[2]
b2 <- coef(model)[3]
cat(sprintf("The logistic regression model is: p = 1 / (1 + exp(-(%f + %f*x1 + %f*x2)))\n", b0, b1, b2))

# Predict probabilities
y_prob <- predict(model, newdata = test_data, type = "response")

# Apply the classification threshold
y_pred <- ifelse(y_prob >= r, 1, 0)

# Calculate misclassification rate
misclassification_rate <- mean(test_data$y != y_pred)
cat(sprintf("Misclassification rate: %f\n", misclassification_rate))

# Plotting
ggplot(test_data, aes(x = x1, y = x2, color = as.factor(y_pred))) +
  geom_point(alpha = 0.5) +
  scale_color_manual(values = c('0' = 'blue', '1' = 'red'), labels = c('0' = 'No Diabetes', '1' = 'Diabetes')) +
  labs(title = 'Plasma Glucose Concentration vs Age Colored by Predicted Diabetes Level',
       x = 'Plasma Glucose Concentration',
       y = 'Age',
       color = 'Predicted Diabetes Level') +
  theme_minimal() +
  theme(legend.position = "bottom")

##################################Task 3##################################
library(ggplot2)
library(dplyr)
library(reshape2)

# Create a meshgrid for the contour plot
x1_range <- range(df$Plasma_glucose_concentration)
x2_range <- range(df$Age)

x1_seq <- seq(from = x1_range[1] - 1, to = x1_range[2] + 1, by = 0.1)
x2_seq <- seq(from = x2_range[1] - 1, to = x2_range[2] + 1, by = 0.1)

grid <- expand.grid(x1 = x1_seq, x2 = x2_seq)

# Predict probabilities on the meshgrid
grid$prob <- predict(model, newdata = grid, type = "response")


# Reshape for ggplot
grid_melt <- melt(grid, id.vars = c("x1", "x2"))

# Plotting
ggplot() +
  geom_tile(data = grid_melt, aes(x = x1, y = x2), alpha = 0.5) +
  scale_fill_gradient2(data = grid_melt, low = "blue", high = "red", mid = "white", midpoint = r, limit = c(0, 1), space = "Lab", name = "Probability") +
  geom_contour(data = grid_melt, aes(x = x1, y = x2, z = y_prob), breaks = c(r), color = "grey") +
  geom_point(data = test_data, aes(x = x1, y = x2, color = as.factor(y_pred)), alpha = 0.5) +
  labs(title = 'Plasma Glucose Concentration vs Age with Decision Boundary',
       x = 'Plasma Glucose Concentration',
       y = 'Age') +
  theme_minimal() +
  theme(legend.position = "bottom")

##################################Task 4##################################
library(ggplot2)	
library(dplyr)

# Predicting with a lower threshold r = 0.2
lower_threshold <- 0.2
test_data$y_pred_low_threshold <- ifelse(predict(model, newdata = test_data, type = "response") >= lower_threshold, 1, 0)

# Plot for lower threshold
ggplot(test_data, aes(x = x1, y = x2, color = as.factor(y_pred_low_threshold))) +
  geom_point(alpha = 0.5) +
  scale_color_manual(values = c('0' = 'blue', '1' = 'red')) +
  labs(title = 'Plasma Glucose Concentration vs Age with r = 0.2',
       x = 'Plasma Glucose Concentration',
       y = 'Age',
       color = 'Predicted Class with r = 0.2') +
  theme_minimal() +
  theme(legend.position = "bottom")

# Predicting with a higher threshold r = 0.8
higher_threshold <- 0.8
test_data$y_pred_high_threshold <- ifelse(predict(model, newdata = test_data, type = "response") >= higher_threshold, 1, 0)

# Plot for higher threshold
ggplot(test_data, aes(x = x1, y = x2, color = as.factor(y_pred_high_threshold))) +
  geom_point(alpha = 0.5) +
  scale_color_manual(values = c('0' = 'blue', '1' = 'red')) +
  labs(title = 'Plasma Glucose Concentration vs Age with r = 0.8',
       x = 'Plasma Glucose Concentration',
       y = 'Age',
       color = 'Predicted Class with r = 0.8') +
  theme_minimal() +
  theme(legend.position = "bottom")

##################################Task 5##################################
library(ggplot2)
library(caret)
library(dplyr)

# Assuming 'df' is your DataFrame and it already includes 'x1' and 'x2'
df$z1 <- df$Plasma_glucose_concentration^4
df$z2 <- df$Plasma_glucose_concentration^3 * df$Age
df$z3 <- df$Plasma_glucose_concentration^2 * df$Age^2
df$z4 <- df$Plasma_glucose_concentration * df$Age^3
df$z5 <- df$Age^4

# Define the features and the target variable
X <- df[, c('Plasma_glucose_concentration', 'Age', 'z1', 'z2', 'z3', 'z4', 'z5')]
y <- df$Diabetes

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
ggplot(as.data.frame(X_train), aes(x = Plasma_glucose_concentration, y = Age, color = as.factor(y_train_pred_class))) +
  geom_point(alpha = 0.5) +
  scale_color_manual(values = c('0' = 'blue', '1' = 'red')) +
  labs(title = 'Plasma Glucose Concentration vs Age with Polynomial Features',
       x = 'Plasma Glucose Concentration',
       y = 'Age',
       color = 'Predicted Class') +
  theme_minimal() +
  theme(legend.position = "bottom")

