# 数据导入
rm(list=ls())
filename1 <- "E:/大三上/2、机器学习/期末论文/train.csv"
data_train <- read.csv(filename1, header=T, na.strings=c("NA"))
filename2 <- "E:/大三上/2、机器学习/期末论文/test.csv"
data_test <- read.csv(filename2, header=T, na.strings=c("NA"))


# 数据分析
library(dplyr)
library(ggplot2)
library(gridExtra)
# 房屋状况
aggregate(data_train$label,by=list(region=data_train$house_ownership),length)
a1<-ggplot(data = data_train) + 
  geom_bar(mapping = aes(x = house_ownership, fill = label), position = "fill")
rent1 <- filter(data_train, label==1, house_ownership=='rented')
own1 <- filter(data_train, label==1, house_ownership=='owned')
on1 <- filter(data_train, label==1, house_ownership=='norent_noown')
# 汽车状况
aggregate(data_train$label,by=list(region=data_train$car_ownership),length)
a2<-ggplot(data = data_train) + 
  geom_bar(mapping = aes(x = label, fill = car_ownership), position = "fill")
carno1 <- filter(data_train, label==1, car_ownership=='no')
caryes1 <- filter(data_train, label==1, car_ownership=='yes')
# 婚姻状况
aggregate(data_train$label,by=list(region=data_train$is_married),length)
a3<-ggplot(data = data_train) + 
  geom_bar(mapping = aes(x = label, fill = is_married), position = "fill")

grid.arrange(a3,a1,a2,ncol=3)

single1 <- filter(data_train, label==1, is_married=='single')
marry1 <- filter(data_train, label==1, is_married=='married')
# 收入
ggplot(data = data_train, 
       mapping = aes(x = income, y = ..density..)) + 
  geom_freqpoly(mapping = aes(colour = label), 
                binwidth = 1000000)
# 年龄
ggplot(data = data_train, 
             mapping = aes(x = age, y = ..density..)) + 
  geom_freqpoly(mapping = aes(colour = label), 
                binwidth = 10)
# 工作年限
ggplot(data = data_train, 
       mapping = aes(x = experience_years, y = ..density..)) + 
  geom_freqpoly(mapping = aes(colour = label), 
                binwidth = 3)
# 现在房屋拥有时间
ggplot(data = data_train, 
       mapping = aes(x = current_house_years, y = ..density..)) + 
  geom_freqpoly(mapping = aes(colour = label), 
                binwidth = 1)

label1 <- filter(data_train, label == 1)
label0 <- filter(data_train, label == 0)
# 地区
aggregate(label1$label,by=list(region=label1$region),length)
# 城市
aggregate(label1$label,by=list(region=label1$city),length)
# 职业
aggregate(label1$label,by=list(region=label1$profession),length)




# 数据预处理
# 缺失值
colSums(is.na(data_train))



# 1. lightgbm
library(caret)
library(lightgbm) 
library(data.table)
library(Matrix)
library(MLmetrics)
library(pROC)
# 类型转化
data_train$label <- as.factor(data_train$label)
data_train[data_train$is_married=="married","is_married"] <- 0
data_train[data_train$is_married=="single","is_married"] <- 1
data_train$is_married <- as.factor(data_train$is_married)
data_train[data_train$house_ownership=="norent_noown","house_ownership"] <- 0
data_train[data_train$house_ownership=="owned","house_ownership"] <- 1
data_train[data_train$house_ownership=="rented","house_ownership"] <- 2
data_train$house_ownership <- as.factor(data_train$house_ownership)
data_train[data_train$car_ownership=="no","car_ownership"] <- 0
data_train[data_train$car_ownership=="yes","car_ownership"] <- 1
data_train$car_ownership <- as.factor(data_train$car_ownership)

data_test$label <- as.factor(data_test$label)
data_test[data_test$is_married=="married","is_married"] <- 0
data_test[data_test$is_married=="single","is_married"] <- 1
data_test$is_married <- as.factor(data_test$is_married)
data_test[data_test$house_ownership=="norent_noown","house_ownership"] <- 0
data_test[data_test$house_ownership=="owned","house_ownership"] <- 1
data_test[data_test$house_ownership=="rented","house_ownership"] <- 2
data_test$house_ownership <- as.factor(data_test$house_ownership)
data_test[data_test$car_ownership=="no","car_ownership"] <- 0
data_test[data_test$car_ownership=="yes","car_ownership"] <- 1
data_test$car_ownership <- as.factor(data_test$car_ownership)


a = c(-1,-13)
train_x = as.matrix(data_train[, a])
train_y = as.matrix(data_train[, 13])
test_x = as.matrix(data_test[, a])
test_y = as.matrix(data_test[, 13])

cnames1 <- c("income","age","experience_years","is_married","city","region","current_job_years",  
            "current_house_years","house_ownership","car_ownership","profession")
cnames2 <- c("label")
rnames1 <- c(1:168000)
rnames2 <- c(1:84000)
train_x <- lapply(train_x,as.numeric)
train_x <- matrix(data = train_x, nrow = 168000, ncol = 11, byrow = FALSE, dimnames = list(rnames1,cnames1))
train_y <- lapply(train_y,as.numeric)
train_y <- matrix(data = train_y, nrow = 168000, ncol = 1, byrow = FALSE, dimnames = list(rnames1,cnames2))
test_x <- lapply(test_x,as.numeric)
test_x <- matrix(data = test_x, nrow = 84000, ncol = 11, byrow = FALSE, dimnames = list(rnames2,cnames1))
test_y <- lapply(test_y,as.numeric)
test_y <- matrix(data = test_y, nrow = 84000, ncol = 1, byrow = FALSE, dimnames = list(rnames2,cnames2))

dtrain = lgb.Dataset(train_x, label = train_y)
dtest = lgb.Dataset.create.valid(dtrain, data=test_x, label = test_y)

valids = list(test = dtest)

lgb_params <- list(
  n_estimators = 10000,
  learning_rate = 0.1,
  max_depth = 9,
  num_leaves = 29,
  subsample = 0.7,
  metric = 'auc',
  boosting_type = 'gbdt',
  objective =  'binary'
)


lgb_cv.score.5 <- rep(0, 5)
train_y = as.matrix(data_train[, 13])
lgb_folds <- createDataPartition(y=train_y, times = 5, p = 0.2, list = TRUE, groups = min(5, length(train_y)))
lgb_pred_test = 0
for (i in 1:5) {
  train_x_train = train_x[-lgb_folds[[i]],]
  train_y_train = train_y[-lgb_folds[[i]],]
  train_x_valid = train_x[lgb_folds[[i]],]
  train_y_valid = train_y[lgb_folds[[i]],]
  dtrain_train = lgb.Dataset(train_x_train, label = train_y_train)
  dtest_valid = lgb.Dataset.create.valid(dtrain_train, data=train_x_valid, label = train_y_valid)
  valids = list(valid = dtest_valid)
  lgb_model = lgb.train(lgb_params,
                    dtrain_train,
                    nrounds = 100,
                    valids,
                    verbose = 1L,
                    early_stopping_rounds = 200,
                    eval = "binary_logloss")
  print(paste("best valid accuracy:",lgb_model$best_score))
  lgb_cv.score.5[i] <- lgb_model$best_score
  lgb_pred_test = lgb_pred_test + predict(lgb_model, test_x, num_iteration=lgb_model[["best_iter"]])
}
lgb_pred_test = lgb_pred_test / 5

auc(as.numeric(test_y), lgb_pred_test)
roc_lgb<-roc(as.numeric(test_y),lgb_pred_test)
plot(roc_lgb,print.auc=TRUE,auc.polygon=TRUE,grid=c(0.1,0.2),grid.col=c("green","red"),max.auc.polygon=TRUE,auc.polygon.col="skyblue",print.thres=TRUE)

tree_imp = lgb.importance(lgb_model, percentage = T)
lgb.plot.importance(tree_imp, measure = "Gain")

library(yardstick)
library(ggplot2)
# The confusion matrix from a single assessment set (i.e. fold)
lgb_pred_test_matrix <- rep(0,length(test_y))
lgb_pred_test_matrix[lgb_pred_test>.5] <-1
lgb_confmatr <- data.frame(Actual = as.factor(data_test[,13]),
                           Prediction = as.factor(lgb_pred_test_matrix))
lgb_cm <- conf_mat(lgb_confmatr, Actual, Prediction)

autoplot(lgb_cm, type = "heatmap") +
  scale_fill_gradient(low="#D6EAF8",high = "#2E86C1")

# 2.XGboost
library(xgboost)
library(Matrix)

filename1 <- "E:/大三上/2、机器学习/期末论文/train.csv"
data_train <- read.csv(filename1, header=T, na.strings=c("NA"))
filename2 <- "E:/大三上/2、机器学习/期末论文/test.csv"
data_test <- read.csv(filename2, header=T, na.strings=c("NA"))
b <- -1
train_data <- data_train[, b]
test_data <- data_test[, b]

test_matrix <- sparse.model.matrix(label ~ .-1, data = test_data)
test_label <-  as.numeric(test_data$label)
test_fin <- list(data=test_matrix,label=test_label) 
xgb_dtest <- xgb.DMatrix(data = test_fin$data, label = test_fin$label)

xgb_params <- list(
  learning_rate = 0.1,
  n_estimators = 10000,
  max_depth = 5,
  min_child_weight = 3,
  subsample = 0.7,
  objective = 'binary:logistic'
)

xgb_cv.score.5 <- rep(0, 5)
xgb_pred_test = 0
for (i in 1:5) {
  train_data_train = train_data[-lgb_folds[[i]],]
  train_data_valid = train_data[lgb_folds[[i]],]
  train_matrix <- sparse.model.matrix(label ~ .-1, data = train_data_train)
  valid_matrix <- sparse.model.matrix(label ~ .-1, data = train_data_valid)
  train_label <- as.numeric(train_data_train$label)
  valid_label <-  as.numeric(train_data_valid$label)
  train_fin <- list(data=train_matrix,label=train_label) 
  valid_fin <- list(data=valid_matrix,label=valid_label) 
  xgb_dtrain <- xgb.DMatrix(data = train_fin$data, label = train_fin$label) 
  xgb_dvalid <- xgb.DMatrix(data = valid_fin$data, label = valid_fin$label)
  watchlist <- list(train=xgb_dtrain, valid=xgb_dvalid)
  
  xgb_model <- xgb.train(xgb_params,
                         data=xgb_dtrain, 
                         eta=1, 
                         nrounds=4000, 
                         watchlist=watchlist, 
                         eval.metric = "auc", 
                         eval.metric = "logloss",
                         verbose=10,
                         early_stopping_rounds=400)
  xgb_cv.score.5[i] <- xgb_model$best_score
  xgb_pred_test = xgb_pred_test + predict(xgb_model, test_fin$data, num_iteration=xgb_model[["best_iter"]])
}
xgb_pred_test = xgb_pred_test / 5


auc(as.numeric(test_y), xgb_pred_test)
roc_xgb<-roc(as.numeric(test_y),xgb_pred_test)
plot(roc_xgb,print.auc=TRUE,auc.polygon=TRUE,grid=c(0.1,0.2),grid.col=c("green","red"),max.auc.polygon=TRUE,auc.polygon.col="skyblue",print.thres=TRUE)

importance_matrix <- xgb.importance(model = xgb_model)
xgb.plot.importance(importance_matrix = importance_matrix)

# The confusion matrix from a single assessment set (i.e. fold)
xgb_pred_test_matrix <- rep(0,length(test_y))
xgb_pred_test_matrix[xgb_pred_test>.5] <-1
xgb_confmatr <- data.frame(Actual = as.factor(data_test[,13]),
                           Prediction = as.factor(xgb_pred_test_matrix))
xgb_cm <- conf_mat(xgb_confmatr, Actual, Prediction)

autoplot(xgb_cm, type = "heatmap") +
  scale_fill_gradient(low="#D6EAF8",high = "#2E86C1")



# catboost
# 
filename3 <- "E:/大三上/2、机器学习/期末论文/cat.csv"
catb_pred_test <- read.csv(filename3, header=T, na.strings=c("NA"))
catb_pred_test <- as.numeric(as.matrix(catb_pred_test))

auc(as.numeric(test_y), catb_pred_test)
roc_catb<-roc(as.numeric(test_y),catb_pred_test)
plot(roc_catb,print.auc=TRUE,auc.polygon=TRUE,grid=c(0.1,0.2),grid.col=c("green","red"),max.auc.polygon=TRUE,auc.polygon.col="skyblue",print.thres=TRUE)

# The confusion matrix from a single assessment set (i.e. fold)
catb_pred_test_matrix <- rep(0,length(test_y))
catb_pred_test_matrix[catb_pred_test>.5] <-1
catb_confmatr <- data.frame(Actual = as.factor(data_test[,13]),
                           Prediction = as.factor(catb_pred_test_matrix))
catb_cm <- conf_mat(catb_confmatr, Actual, Prediction)

autoplot(catb_cm, type = "heatmap") +
  scale_fill_gradient(low="#D6EAF8",high = "#2E86C1")


# 模型融合
pred_test1 <- (lgb_pred_test + xgb_pred_test + catb_pred_test) / 3
auc(as.numeric(test_y), pred_test1)

pred_test2 <- 0.1 * lgb_pred_test + 0.2 * xgb_pred_test + 0.7 * catb_pred_test
auc(as.numeric(test_y), pred_test2)



