# Install Required Packages
packages <- c(
  "tidyverse", "lubridate", "DataExplorer", "corrplot", "cluster", "factoextra",
  "randomForest", "caret", "forecast", "gridExtra", "ggplot2", "reshape2",
  "naniar", "tseries", "FactoMineR", "pdp", "rpart", "rpart.plot", "ggridges"
)
installed <- packages %in% rownames(installed.packages())
if (any(!installed)) install.packages(packages[!installed])
lapply(packages, library, character.only = TRUE)

# 1. Load and Prepare Dataset

data <- read.csv("marketing_campaign.csv", sep = "\t")
str(data)
data$Dt_Customer <- dmy(data$Dt_Customer)
data <- data %>% select(-ID, -Z_CostContact, -Z_Revenue, -Complain)
data <- data %>% drop_na()

# 2. Feature Engineering

data <- data %>% mutate(
  Age = 2025 - Year_Birth,
  Total_Spend = MntWines + MntFruits + MntMeatProducts +
    MntFishProducts + MntSweetProducts + MntGoldProds,
  Customer_Tenure = as.numeric(difftime(Sys.Date(), Dt_Customer, units = "days"))
)

# 3. Exploratory Data Analysis

summary(data)
glimpse(data)
vis_miss(data)

# Density Plots

plot_density <- function(var) {
  ggplot(data, aes_string(x = var)) +
    geom_density(fill = "skyblue", alpha = 0.5) +
    labs(title = paste("Density Plot of", var), x = var)
}
density_plots <- lapply(names(data %>% select_if(is.numeric)), plot_density)
do.call(grid.arrange, c(density_plots[1:6], ncol = 2))

# Correlation Plot

numeric_vars <- data %>% select_if(is.numeric)
corrplot(cor(numeric_vars), method = "color", type = "upper", tl.cex = 0.8)

# Violin Plot

ggplot(data, aes(x = Education, y = Total_Spend, fill = Education)) +
  geom_violin(trim = FALSE) +
  geom_boxplot(width = 0.1, fill = "white") +
  labs(title = "Violin Plot of Total Spend by Education")

# Ridge Plot

ggplot(data, aes(x = Total_Spend, y = Marital_Status, fill = Marital_Status)) +
  geom_density_ridges(alpha = 0.7) +
  labs(title = "Spend Distribution by Marital Status")

# 4. Outlier Detection

outliers <- boxplot.stats(data$Income)$out
print(outliers)
ggplot(data, aes(x = "", y = Income)) + geom_boxplot() + labs(title = "Income Outliers")

# 5. Clustering (K-Means)

scaled_data <- scale(data %>% select_if(is.numeric))
fviz_nbclust(scaled_data, kmeans, method = "wss")
set.seed(123)
kmeans_model <- kmeans(scaled_data, centers = 4, nstart = 25)
data$Cluster <- as.factor(kmeans_model$cluster)
fviz_cluster(kmeans_model, data = scaled_data)

# PCA for visualization

pca_result <- prcomp(scaled_data)

# Create data frame with first two principal components and cluster labels

pca_df <- data.frame(PC1 = pca_result$x[, 1],
                     PC2 = pca_result$x[, 2],
                     Cluster = factor(kmeans_model$cluster))

# Plot with ggplot2

ggplot(pca_df, aes(x = PC1, y = PC2, color = Cluster)) +
  geom_point(alpha = 0.7, size = 2) +
  labs(
    title = "K-Means Clustering (PCA Projection)",
    x = "Principal Component 1",
    y = "Principal Component 2"
  ) +
  theme_minimal() +
  scale_color_brewer(palette = "Set2")


# 6. Classification (Random Forest with Cross-Validation)

data$Response <- as.factor(data$Response)
set.seed(123)
train_index <- createDataPartition(data$Response, p = 0.7, list = FALSE)
train_data <- data[train_index, ]
test_data <- data[-train_index, ]

ctrl <- trainControl(method = "cv", number = 5)
cv_rf <- train(Response ~ ., data = train_data, method = "rf", trControl = ctrl)
print(cv_rf)

rf_model <- randomForest(Response ~ ., data = train_data, importance = TRUE)
predictions <- predict(rf_model, newdata = test_data)
conf_mat <- confusionMatrix(predictions, test_data$Response)
print(conf_mat)

conf_table <- as.data.frame(conf_mat$table)
colnames(conf_table) <- c("Prediction", "Reference", "Freq")

ggplot(conf_table, aes(x = Reference, y = Prediction, fill = Freq)) +
  geom_tile() +
  geom_text(aes(label = Freq), color = "white", size = 6) +
  scale_fill_gradient(low = "blue", high = "red") +
  labs(title = "Random Forest Confusion Matrix Heatmap") +
  theme_minimal()

importance <- importance(rf_model)
varImpPlot(rf_model)

# 6B. Decision Tree Classification
dt_model <- rpart(Response ~ ., data = train_data, method = "class")
rpart.plot(
  dt_model,
  type = 2,             # more readable labels
  extra = 104,          # to show both % and count at nodes
  under = TRUE,         # to show the split condition under the node
  faclen = 0,           # to not to shorten factor labels
  fallen.leaves = TRUE, # tidy bottom layout
  main = "Decision Tree for Campaign Response"
)

dt_preds <- predict(dt_model, test_data, type = "class")
conf_dt <- confusionMatrix(dt_preds, test_data$Response)
print(conf_dt)

dt_conf_table <- as.data.frame(conf_dt$table)
colnames(dt_conf_table) <- c("Prediction", "Reference", "Freq")

ggplot(dt_conf_table, aes(x = Reference, y = Prediction, fill = Freq)) +
  geom_tile() +
  geom_text(aes(label = Freq), color = "white", size = 6) +
  scale_fill_gradient(low = "lightblue", high = "red") +
  labs(title = "Decision Tree Confusion Matrix Heatmap") +
  theme_minimal()

# 7. Regression Analysis

lm_model <- lm(Total_Spend ~ Income + Age + Kidhome + Teenhome + Education + Marital_Status, data = data)
summary(lm_model)

par(mfrow = c(2, 2))
plot(lm_model)
par(mfrow = c(1, 1))

predicted_spend <- predict(lm_model, data)
actual_spend <- data$Total_Spend
rmse_lm <- sqrt(mean((predicted_spend - actual_spend)^2))
cat("RMSE for Linear Regression: ", rmse_lm)

ggplot(data.frame(Actual = actual_spend, Predicted = predicted_spend), aes(x = Actual, y = Predicted)) +
  geom_point(color = "blue", alpha = 0.5) +
  geom_abline(slope = 1, intercept = 0, color = "red") +
  labs(title = "Actual vs Predicted Spend")

# 8. Time Series Forecasting

enrollment_ts <- data %>%
  mutate(Month = floor_date(Dt_Customer, "month")) %>%
  group_by(Month) %>%
  summarise(Enrollments = n())

ts_data <- ts(enrollment_ts$Enrollments, start = c(2012, 1), frequency = 12)
decomposed_ts <- decompose(ts_data)
plot(decomposed_ts)

ets_model <- ets(ts_data)
forecast_ets <- forecast(ets_model, h = 12)
plot(forecast_ets)

auto_arima_model <- auto.arima(ts_data)
summary(auto_arima_model)
forecast_arima <- forecast(auto_arima_model, h = 12)
plot(forecast_arima)

autoplot(ts_data) +
  autolayer(forecast_ets$mean, series = "ETS", PI = FALSE) +.
  autolayer(forecast_arima$mean, series = "ARIMA", PI = FALSE) +
  labs(title = "ETS vs ARIMA Forecasts", y = "Enrollments") +
  theme_minimal()

# 9. Model Comparison Summary
cat("\n--- MODEL COMPARISON ---\n")
cat("RF Accuracy:", round(conf_mat$overall['Accuracy']*100, 2), "%\n")
cat("DT Accuracy:", round(conf_dt$overall['Accuracy']*100, 2), "%\n")
cat("Regression RMSE:", round(rmse_lm, 2), "\n")
cat("ETS AIC:", round(ets_model$aic, 2), "\n")
cat("ARIMA AIC:", round(auto_arima_model$aic, 2), "\n")
