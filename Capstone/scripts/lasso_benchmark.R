library(data.table)
library(glmnet)
library(caret)
setwd("/Users/caoxiang/Dropbox/Allstate")

### read data
train <- fread('train.csv', header = TRUE)
test <- fread('test.csv', header = TRUE)
loss <- log(train$loss)
combi <- rbind(train[, c('loss', 'id'):=NULL], test[, ('id'):=NULL])


categorical_var <- names(combi)[which(sapply(combi, is.character))]
numeric_var <- names(combi)[which(sapply(combi, is.numeric))]

combi_cat <- combi[, lapply(.SD, as.factor), .SDcols = categorical_var]
combi_con <- combi[, lapply(.SD, as.numeric), .SDcols = numeric_var]
combi <- cbind(combi_cat, combi_con)

train <- combi[1:nrow(train), ]
test <- combi[(nrow(train)+1) : nrow(combi), ]

train <- cbind(train, loss = loss)


### a function to show cross validation mae
ENet_show_cv_performance <- function(data, labelName, predictors, alpha = 0, k=5, seed=1989,
                                     family = 'gaussian', metric = 'mae'){
  cv_mae <- vector(mode = 'logical', length = k)
  mdl <- paste0(labelName, '~', paste0(predictors, collapse = ' + '), sep=' ')
  set.seed(seed)
  folds <- createFolds(data[ , labelName], k=k, list = TRUE)
  for(i in 1:k){
    kf.train <- data[-folds[[i]], ]
    kf.test <- data[folds[[i]], ]
    x <- model.matrix(as.formula(mdl), data = kf.train)
    y <- kf.train[ , labelName]
    fit <- cv.glmnet(x, y, alpha = alpha, family = family, type.measure = metric)
    newx <- model.matrix(as.formula(mdl), data = kf.test)
    pred <- predict(fit, newx = newx, s='lambda.min', type='response')
    pred_mae <- sum(abs(pred - kf.test[, labelName]))/nrow(kf.test)
    cv_mae[i] <- pred_mae
  }
  return(cv_mae)
}


### check the cv mae on train set, 3-folds
labelName <- 'loss'
predictors <- names(train)[! names(train) %in% labelName]

BeginTime <- Sys.time()
result <- ENet_show_cv_performance(as.data.frame(train), labelName, predictors, k=3)
EndTime <- Sys.time()
EndTime - BeginTime


result2 <- ENet_show_cv_performance(as.data.frame(train)[1:1000,], labelName, predictors, k=3)

### fit the final model and predict test set
sample_train <- cbind(train[1:1000, ], loss = loss[1:1000])


set.seed(1999)
x <- model.matrix(loss~., data = train)
BeginTime <- Sys.time() #7:33
lasso.fit <- cv.glmnet(x, y=loss, family = 'gaussian', type.measure = 'mae')
EndTime <- Sys.time()
EndTime - BeginTime

newx <- model.matrix(~., data = test)
pred <- predict(lasso.fit, newx = x)
pred_mae <- sum(abs(loss[1:10000] - pred))/10000

# performance on train set.
pred_train <- predict(lasso.fit, newx = x, s='lambda.min')
sum(abs(exp(pred_train) - loss))/nrow(train) # 1238.705

pred_test <- predict(lasso.fit, newx = newx, s='lambda.min')
pred_test_original <- exp(pred_test)

### 
submission <- fread('sample_submission.csv', header = TRUE)
submission$loss <- pred_test_original
write.csv(submission, 'lasso_log.csv', row.names = FALSE)
