### xgboost starter
library(data.table)
library(glmnet)
library(caret)
library(xgboost)
#setwd("/Users/caoxiang/Dropbox/Allstate")
#setwd("C:/Users/Administrator/Dropbox/Allstate")

### read data
train <- fread('train.csv', header = TRUE)
test <- fread('test.csv', header = TRUE)
loss <- train$loss
combi <- rbind(train[, c('loss', 'id'):=NULL], test[, ('id'):=NULL])


categorical_var <- names(combi)[which(sapply(combi, is.character))]
numeric_var <- names(combi)[which(sapply(combi, is.numeric))]

combi_cat <- combi[, lapply(.SD, as.factor), .SDcols = categorical_var]
combi_con <- combi[, lapply(.SD, as.numeric), .SDcols = numeric_var]
combi <- cbind(combi_cat, combi_con)

train <- combi[1:nrow(train), ]
test <- combi[(nrow(train)+1) : nrow(combi), ]

#train <- cbind(train, loss = loss)

##########################
train_matrix <- xgb.DMatrix(sparse.model.matrix(~.-1, data = as.data.frame(train)), label=log(loss))
test_matrix <- xgb.DMatrix(sparse.model.matrix(~.-1, data = as.data.frame(test)))
#label <- loss
##########################
set.seed(12345678)
params <- list(
  eta = 0.0404096, # Santander overfitting magic number X2
  gamma = 0,
  max_depth = 6,
  min_child_weight = 1,
  subsample = 0.7,
  colsample_bytree = 0.7,
  alpha = 0,
  objective='reg:linear',
  eval_metric = 'mae',
  num_parallel_tree = 1) 


xgb_cv <- xgb.cv(params = params,
                 train_matrix,
                 nthread = 4,
                 nrounds = 70,
                 nfold = 4,
                 verbose = 1,
                 print_every_n = 10,
                 early_stopping_rounds = 25)

best_nround <- xgb_cv$best_iteration

set.seed(12345)
xgb_fit <- xgb.train(params = params,
                   train_matrix,
                   nthread = 4,
                   nrounds = 1265,
                   verbose = 1,
                   watchlist = list(train=train_matrix),
                   print_every_n = 10
                   )


pred_xgb <- predict(xgb_fit, test_matrix)

submission <- fread('sample_submission.csv', header = TRUE)
submission$loss <- exp(pred_xgb)
write.csv(submission, 'xgb_start_n1265_log.csv', row.names = FALSE)
