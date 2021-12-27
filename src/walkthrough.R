rm(list = ls()); gc()
library(tidyverse)
library(xgboost)

# 讀入函數檔
source("funpack.R")

###### Set Dummy Data ######
n <- 10000
data <- tibble(
  x_1 = rnorm(n, 3.05, 0.35), 
  x_2 = rnorm(n, -4.1, 1.5),
  y = 3.1 * x_1 + 0.5 * x_2 + 0.1 * rnorm(n, 0, 1) + 19
) 

# col選取訓練特徵的欄位
col <- data %>%
  select(-y) %>%
  colnames()

# 切割測試集與訓練集
index <- sample(1:nrow(data), 0.2 * nrow(data))
# training set
trainData <- data[-index, ]
testData <- data[index, ]
# validation set
valIndex <- sample(1:nrow(trainData), 0.2 * nrow(trainData))
valData <- trainData[valIndex, ]
trainData <- trainData[-valIndex, ]

# XGBoost的參數
learningRate <- 0.1 # eta
iter_rounds <- 1000 # 迭代次數
early_stopping_rounds <- 10 # 提早停止訓練次數
verbose <- FALSE # 是否要顯示訓練過程


mu_init <- mean(trainData$y)
sigma_init <- log(sd(trainData$y))

dtrainMu <- xgb.DMatrix(trainData %>% select(col) %>% as.matrix(), label = trainData$y)
dvalMu <- xgb.DMatrix(valData %>% select(col) %>% as.matrix(), label = valData$y)
muModelList <- list()

dtrainSigma <- xgb.DMatrix(trainData %>% select(col) %>% as.matrix(), label = trainData$y)
dvalSigma <- xgb.DMatrix(valData %>% select(col) %>% as.matrix(), label = valData$y)
sigmaModelList <- list()

f_mu.predict <- c(rep(mu_init, nrow(trainData)))
f_sigma.predict <- c(rep(sigma_init, nrow(trainData)))


# Model III Loss
MuLoss <<- function(preds, dtrain, sig = f_sigma.predict) {
  labels <- getinfo(dtrain, "label")
  preds <- preds
  grad <- -((labels - preds) / (1 + (exp(2 * sig) * 0.00001)))
  hess <- (labels - labels + 1)
  return(list(grad = grad, hess = hess))
}
SigmaLoss <<- function(preds, dtrain, mu = f_mu.predict) {
  labels <- getinfo(dtrain, "label")
  preds <- preds
  grad <- -((((labels - mu)^2) / (2 * exp(2 * preds))) - 0.5)
  hess <- (labels - labels + 1)
  return(list(grad = grad, hess = hess))
}
SigmaError <- function(preds, dtrain) {
  labels <- getinfo(dtrain, "label")
  err <- mean(exp(preds))
  return(list(metric = "mean(sigma)", value = err))
}
paramMu <- list(
  max_depth = 10, eta = learningRate,
  objective = MuLoss3, eval_metric = "rmse", min_child_weight = 0
)
paramSigma <- list(
  max_depth = 10, eta = learningRate,
  objective = SigmaLoss3, eval_metric = SigmaError, min_child_weight = 0
)


best_log_eval <- Inf
mu.list <- c(rep(mu_init, nrow(trainData)))
sigma.list <- c(rep(sigma_init, nrow(trainData)))
for (m in c(1:iter_rounds)) {
  # 分成mu跟sigma兩種模型訓練
  f_mu <- xgb.train(
    params = paramMu,
    data = dtrainMu,
    nrounds = 1,
    watchlist = list(train = dtrainMu, eval = dvalMu),
    verbose = FALSE,
    base_score = ifelse(m == 1, mu_init, 0)
  )

  f_sigma <- xgb.train(
    params = paramSigma,
    data = dtrainSigma,
    nrounds = 1,
    watchlist = list(train = dtrainSigma, eval = dvalSigma),
    verbose = FALSE,
    base_score = ifelse(m == 1, sigma_init, 0)
  )

  f_mu.predict <- predict(f_mu, dtrainMu, outputmargin = TRUE)
  f_sigma.predict <- predict(f_sigma, dtrainSigma, outputmargin = TRUE)

  eval_score <- LogScore(
    y = valData$y,
    mu = predict(f_mu, dvalMu),
    sigma = exp(predict(f_sigma, dvalSigma)) + 1e-10
  ) %>% sum()

  train_score <- LogScore(
    y = trainData$y,
    mu = f_mu.predict,
    sigma = exp(f_sigma.predict) + 1e-10
  ) %>% sum()

  # 將預測值為下一次迭代預測值的基準（否則會只有gradient)
  setinfo(dtrainMu, "base_margin", predict(f_mu, dtrainMu, outputmargin = TRUE))
  setinfo(dvalMu, "base_margin", predict(f_mu, dvalMu, outputmargin = TRUE))
  setinfo(dtrainSigma, "base_margin", predict(f_sigma, dtrainSigma, outputmargin = TRUE))
  setinfo(dvalSigma, "base_margin", predict(f_sigma, dvalSigma, outputmargin = TRUE))
  # 儲存模型
  muModelList[[m]] <- f_mu
  sigmaModelList[[m]] <- f_sigma
  
  # early stopping
  if (best_eval > f_mu$evaluation_log$eval_rmse & best_log_eval > eval_score) {
    best_eval <- f_mu$evaluation_log$eval_rmse
    best_log_eval <- eval_score
    best_iter <- m
    cat("Iter:", best_iter, "train_-logP =", train_score, "eval_-logP =", best_log_eval, "\n")
  } else if ((m - best_iter) >= early_stopping_rounds) {
    m_stop <- best_iter
    cat("Iter:", m, "train_-logP =", train_score, "eval_-logP =", best_log_eval, "\n early_stopping: ", m, "->", best_iter, "rounds \n")
    break
  }
  mu.list <- cbind(mu.list, f_mu.predict)
  sigma.list <- cbind(sigma.list, f_sigma.predict)
}

cat("Model III stop at ", best_iter, "round \n")
Model_3_mu.list <- mu.list[, 1:best_iter]
Model_3_sigma.list <- sigma.list[, 1:best_iter]
Model_3_muModelList <- muModelList
Model_3_sigmaModelList <- sigmaModelList
Model_3_m_stop <- best_iter

= model_3_sigma)))