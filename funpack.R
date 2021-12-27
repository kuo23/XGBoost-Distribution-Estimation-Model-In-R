#Functions for NGBoost

# Predict
ModelPredict = function(modelList = modelList, m = m, dtest = dtest){
  pre_table = matrix(ncol = m, nrow = nrow(dtest))
  for (n in 1:m){
    pre_table[, n]= as.array(predict(modelList[[n]], dtest))
  }
  value = (t(matrix(1, nrow = 1, ncol = m) %*% t(pre_table)) %>% as.array())
  return(value)
}

rmse = function(y, y_hat){
  sqrt(mean((y - y_hat)^2))
}

# Plot
NGplot = function(data, m_stop = m_stop, mu.list = mu.list, sigma.list = sigma.list){
  data %>% 
    dplyr::select(y) %>% 
    cbind(., mu.list) %>% 
    `colnames<-`(c("y",c(0, 1:(m_stop-1)))) %>% 
    arrange(y) %>% 
    mutate(i = 1:n()) %>% 
    gather(key = "m", value = "mu", c(-y, -i)) %>% 
    left_join(trainData %>% 
                dplyr::select(y) %>% 
                cbind(., sigma.list) %>% 
                `colnames<-`(c("y",c(0, 1:(m_stop-1)))) %>% 
                arrange(y) %>% 
                mutate(i = 1:n()) %>% 
                gather(key = "m", value = "sigma_log",c(-y, -i)) %>% dplyr::select(-i), by = c("y" = "y", "m"="m")) %>% 
    mutate(m = as.numeric(m)) %>% 
    group_by(m) %>% 
    mutate(fixed_sigma = sd(y-mu))
}

LossPlot = function(plot_df){
  name = deparse(substitute(plot_df))
  plot_df = plot_df %>% mutate(n_round = as.factor(1:n())) %>% 
    gather(key = "gen", value = "value", ori, model_1, model_2 ,model_3 )
  p=ggpubr::ggarrange(ggplot(plot_df , aes(x = gen, y = value, fill = gen))+
                        geom_boxplot(size=1)+
                        labs(title= name)+
                        theme_bw()+ 
                        scale_fill_jama(),
                      # scale_fill_manual(values=wes_palette(n=4, name="Darjeeling1")),
                      ggplot(plot_df, aes(x = n_round, y = value, group = n_round, color = gen)) +
                        geom_point(size=2)+
                        geom_line(color = "lightblue")+
                        theme_bw()+
                        theme(axis.text.x = element_blank())+ 
                        scale_color_jama(),
                      # scale_color_manual(values=wes_palette(n=4, name="Darjeeling1"))
                      ncol = 1, nrow = 2)
  return(p)
}

# Tuning by using cv
ParamCV = function(dtrain, paramTable, nfold=5){
  # paramTable <- expand.grid(eta = c( 0.5, 0.7), max_depth = c(3, 5, 7, 10),
  #                           subsample = c(0.5, 0.7, 1), colsample_bytree = c(0.7, 1))
  cvOutput = NULL
  for(yi in 1:nrow(paramTable)){
    
    # cat(yi, "/", nrow(paramTable), "\n")
    param = paramTable %>%  filter(row_number()==yi)
    origin = xgb.cv(params = param, 
                    data = dtrain, 
                    nround = iter_rounds,
                    nfold = nfold,
                    watchlist = list(train = dtrain), 
                    early_stopping_rounds = 20,
                    verbose = verbose
    )
    
    cvOutput = cvOutput %>% 
      bind_rows(tibble(yi = yi,
                       rmse = origin$evaluation_log$test_rmse_mean[origin$best_iteration], 
                       iter_rounds = origin$best_iteration))
    
  }
  param.cv = paramTable %>% 
    filter(row_number() == cvOutput %>% dplyr::slice(which.min(rmse)) %>% pull(yi)) %>% 
    as.list()
  return(list(param.cv = param.cv, 
              best_iter_rounds = cvOutput$iter_rounds[cvOutput %>% dplyr::slice(which.min(rmse)) %>%
                                                        pull(yi)])
  )
}

LogScore = function(y, mu, sigma){
  score = 0.5*log(sigma^2)+(((y-mu)^2)/(2*sigma^2))
  return(score)
}

ModelI = function(trainData, col ,label, iter_rounds, valData=NULL, valLabel = valLabel, param = NULL){
  
  mu_init = mean(label)
  sigma = sd(label - mu_init)
  
  
  dtrainIter = xgb.DMatrix(trainData %>% select(col) %>% as.matrix(), label = label)
  modelList = list()
  
  CustomLossIter <- function(preds, dtrain, sig2 = sigma^2) {
    labels <- getinfo(dtrain, "label")
    preds <- preds
    grad <- -(1 / sig2) * (labels - preds)
    hess <- 1 / (labels - labels + sig2)
    return(list(grad = grad, hess = hess))
  }
  
  
  if(is.null(param)){
    paramCustIter <- list(
      max_depth = 10, eta = learningRate, 
      objective = CustomLossIter , eval_metric = "rmse", min_child_weight = 0
    )
    
  }else if(is.null(param$objective) & is.null(param$eval_metric)){
    paramMu <- c(param,
                 list(objective = CustomLossIter,
                      eval_metric = "rmse"))
    
  }else if (!is.null(param$objective) | !is.null(param$eval_metric)){
    param$objective = NULL
    param$eval_metric = NULL
    paramCustIter<- c(param,
                      list(objective = CustomLossIter,
                           eval_metric = "rmse"))
  }
  
  # Save every sigma during training
  sigList = rep(sigma, nrow(trainData))
  mulist = rep(mean(label), nrow(trainData))
  best_eval = Inf
  best_log_eval = Inf
  if (!is.null(valData)){
    dvalIter <- xgb.DMatrix(valData %>% select(col) %>% as.matrix(), label = valLabel)
    
    for(i in c(1:iter_rounds)){
      # cat("第", i, "次，sigma =", sigma, "\n")
      
      # start with base_score in  the first round
      bst <- xgb.train(params = paramCustIter, 
                       data = dtrainIter, 
                       nrounds = 1, 
                       watchlist = list(train = dtrainIter, eval = dvalIter), 
                       verbose = verbose, 
                       base_score = ifelse(i==1, mu_init, 0)
      )
      
      
      # predict output each round
      bst.predict = predict(bst, dtrainIter, outputmargin = TRUE)
      sigma = (sd(label - bst.predict))
      eval_score = LogScore(y = valLabel, 
                            mu = predict(bst, dvalIter), 
                            sigma = (sd(valLabel - predict(bst, dvalIter)))) %>% mean()
      train_score =  LogScore(y = label, 
                              mu = bst.predict, 
                              sigma = sigma) %>% mean()
      # set predict value as initial value next round
      setinfo(dtrainIter, "base_margin", predict(bst, dtrainIter, outputmargin = TRUE))
      setinfo(dvalIter, "base_margin", predict(bst, dvalIter, outputmargin = TRUE))
      # set sigma value as residual at each round
      # sigma = 1
      
      
      # sigma = rmse(label, bst.predict)^2
      modelList[[i]] = bst
      
      sigList = cbind(sigList, rep(sigma, nrow(trainData)))
      mulist = cbind(mulist, bst.predict)
      # early stopping
      if(best_eval > bst$evaluation_log$eval_rmse & best_log_eval > eval_score) {
        best_eval = bst$evaluation_log$eval_rmse
        best_iter = i
      }else if((i - best_iter) >= early_stopping_rounds){
        i_stop = best_iter
        best_eval = modelList[[i_stop]]$evaluation_log$eval_rmse
        break
      }
      
    }
    cat("Model stop at ", best_iter, "round \n")
    model_1 = modelList
    model_1_best_iter = i-early_stopping_rounds
    return(list(model_1 = modelList,
                mu.list = mulist[, 1:best_iter],
                sigma.list=sigList[, 1:best_iter],
                m_stop = model_1_best_iter))
    
  }else if(is.null(valData)){
    
    for(i in c(1:iter_rounds)){
      # cat("第", i, "次，sigma =", sigma, "\n")
      
      # start with base_score in  the first round
      bst <- xgb.train(params = paramCustIter, 
                       data = dtrainIter, 
                       nrounds = 1, 
                       watchlist = list(train = dtrainIter), 
                       verbose = verbose, 
                       base_score = ifelse(i==1, mu_init, 0)
      )
      
      
      # predict output each round
      bst.predict = predict(bst, dtrainIter, outputmargin = TRUE)
      sigma = (sd(label - bst.predict))
      
      train_score =  LogScore(y = label, 
                              mu = bst.predict, 
                              sigma = sigma) %>% mean()
      # set predict value as initial value next round
      setinfo(dtrainIter, "base_margin", predict(bst, dtrainIter, outputmargin = TRUE))
      setinfo(dvalIter, "base_margin", predict(bst, dvalIter, outputmargin = TRUE))
      # set sigma value as residual at each round
      # sigma = 1
      
      
      # sigma = rmse(label, bst.predict)^2
      modelList[[i]] = bst
      sigList = cbind(sigList, rep(sigma, nrow(trainData)))
      mulist = cbind(mulist, bst.predict)
      # early stopping
      if(best_eval > bst$evaluation_log$train_rmse & best_log_eval > train_score) {
        best_iter = i
      }else if((i - best_iter) >= early_stopping_rounds){
        i_stop = best_iter
        break
      }
      
    }
    cat("Model stop at ", best_iter, "round \n")
    model_1 = modelList
    model_1_best_iter = i-early_stopping_rounds
    return(list(model_1 = modelList,
                mu.list = mulist[, 1:best_iter],
                sigma.list=sigList[, 1:best_iter],
                m_stop = model_1_best_iter))
  }
}

NGBoost = function(trainData, col, label, iter_rounds, valData=NULL, valLabel=NULL, param = NULL, MuLoss = NULL, SigmaLoss = NULL){
  
  mu_init = mean(label)
  sigma_init = log(sd(label))
  
  dtrainMu = xgb.DMatrix(trainData %>% select(col)  %>% as.matrix(), label = label)
  dtrainSigma = xgb.DMatrix(trainData %>% select(col) %>% as.matrix(), label = label)
  
  muModelList = list()
  sigmaModelList = list()
  f_mu.predict <<- c(rep(mu_init, nrow(trainData)))
  f_sigma.predict <<- c(rep(sigma_init, nrow(trainData)))
  
  # Parameter Loss
  if(is.null(MuLoss)){
    MuLoss <- function(preds, dtrain, sig = f_sigma.predict) {
      labels <- getinfo(dtrain, "label")
      preds <- preds
      # grad <- -(labels - preds)
      grad <- -((labels - preds)/(1+(exp(2*sig)*0.00001)))
      # grad <- -((labels - preds)/(exp(2*sig)+1e-10))
      # hess <-  1 / (labels - labels + exp(2*sig)+1e-10)
      hess <- (labels - labels + 1)
      return(list(grad = grad, hess = hess))
    }
    
  }
  
  if(is.null(SigmaLoss)){
    SigmaLoss <- function(preds, dtrain, mu = f_mu.predict) {
      labels <- getinfo(dtrain, "label")
      preds <- preds
      # grad <- 1-(((labels - mu)^2)/(exp(2 * preds)+ 1e-10))
      grad <- -((((labels - mu)^2)/(2 * exp(2 * preds))) - 0.5)
      # hess <- 2*((labels - mu)^2)/(exp(2 * preds)+1e-10)
      hess <-(labels - labels + 1)
      return(list(grad = grad, hess = hess))
    }
  }
  
  # Parameter Error
  SigmaError <- function(preds, dtrain) {
    labels <- getinfo(dtrain, "label")
    err <- mean(exp(preds))
    return(list(metric = "mean(sigma)", value = err))
  }
  
  if(is.null(param)){
    paramMu <- list(
      max_depth = 10, eta = learningRate, 
      objective = MuLoss , eval_metric = "rmse", min_child_weight = 0
    )
    
    paramSigma <- list(
      max_depth = 10, eta = learningRate, 
      objective = SigmaLoss , eval_metric = SigmaError, min_child_weight = 0
    )
  }else if(is.null(param$objective) & is.null(param$eval_metric)){
    paramMu <- c(param,
                 list(objective = MuLoss,
                      eval_metric = "rmse"))
    
    paramSigma <- c(param, list(
      objective = SigmaLoss , eval_metric = SigmaError
    ))
    
  }else if (!is.null(param$objective) | !is.null(param$eval_metric)){
    param$objective = NULL
    param$eval_metric = NULL
    paramMu <- c(param,
                 list(objective = MuLoss,
                      eval_metric = "rmse"))
    
    paramSigma <- c(param, list(
      objective = SigmaLoss , eval_metric = SigmaError, min_child_weight = 0
    ))
    
  }
  
  # Save every sigma during training
  best_eval = Inf
  best_log_eval = Inf
  mu.list= c(rep(mu_init, nrow(trainData)))
  sigma.list= c(rep(sigma_init, nrow(trainData)))
  if (!is.null(valData)){
    dvalMu = xgb.DMatrix(valData %>% select(col)%>% as.matrix(), label = valLabel)
    dvalSigma = xgb.DMatrix(valData %>% select(col)%>% as.matrix(), label =valLabel)
    
    for(m in c(1:iter_rounds)){
      # start with base_score in  the first round
      f_mu <- xgb.train(params = paramMu, 
                        data = dtrainMu, 
                        nrounds = 1, 
                        watchlist = list(train = dtrainMu, eval = dvalMu),
                        verbose = verbose, 
                        base_score = ifelse(m==1, mu_init, 0)
                        
      )
      f_sigma <- xgb.train(params = paramSigma, 
                           data = dtrainSigma, 
                           nrounds = 1, 
                           watchlist = list(train = dtrainSigma, eval = dvalSigma),
                           verbose = verbose, 
                           base_score = ifelse(m==1, sigma_init, 0)
                           
      )
      
      # predict output each round
      f_mu.predict <<- predict(f_mu, dtrainMu, outputmargin = TRUE)
      f_sigma.predict <<- predict(f_sigma, dtrainSigma, outputmargin = TRUE)
      
      eval_score = LogScore(y = valLabel, 
                            mu = predict(f_mu, dvalMu), 
                            sigma = exp(predict(f_sigma, dvalSigma))+1e-10) %>% mean()
      train_score =  LogScore(y = label, 
                              mu = f_mu.predict, 
                              sigma = exp(f_sigma.predict)+1e-10) %>% mean()
      # set predict value as initial value next round
      setinfo(dtrainMu, "base_margin", predict(f_mu,dtrainMu, outputmargin = TRUE))
      setinfo(dvalMu, "base_margin", predict(f_mu, dvalMu, outputmargin = TRUE))
      setinfo(dtrainSigma, "base_margin", predict(f_sigma,dtrainSigma, outputmargin = TRUE))
      setinfo(dvalSigma, "base_margin", predict(f_sigma, dvalSigma, outputmargin = TRUE))
      muModelList[[m]] = f_mu
      sigmaModelList[[m]] = f_sigma
      
      # early stopping
      
      if(best_eval > f_mu$evaluation_log$eval_rmse & best_log_eval > eval_score){
        best_eval = f_mu$evaluation_log$eval_rmse
        best_log_eval = eval_score
        best_iter = m
        cat("Iter:", best_iter,"train_-logP =",train_score, "eval_-logP =", best_log_eval, "\n")
      }else if((m - best_iter) >= early_stopping_rounds){
        m_stop = best_iter
        cat("Iter:", best_iter,"train_-logP =",train_score, "eval_-logP =", best_log_eval, "early_stopping: ", m, "->", best_iter, "rounds \n")
        break
      }
      mu.list = cbind(mu.list, f_mu.predict)
      sigma.list = cbind(sigma.list, f_sigma.predict)
    }
    cat("Model stop at ", best_iter, "round \n")
    mu.list = mu.list[,1 : best_iter]
    sigma.list = sigma.list[,1 : best_iter]
    return(list(muModelList=muModelList,
                sigmaModelList =sigmaModelList,
                mu.list = mu.list,
                sigma.list=sigma.list,
                m_stop = best_iter))
    
  }else if(is.null(valData)){
    
    for(m in c(1:iter_rounds)){
      # start with base_score in  the first round
      f_mu <- xgb.train(params = paramMu, 
                        data = dtrainMu, 
                        nrounds = 1, 
                        watchlist = list(train = dtrainMu),
                        verbose = verbose, 
                        base_score = ifelse(m==1, mu_init, 0)
                        
      )
      f_sigma <- xgb.train(params = paramSigma, 
                           data = dtrainSigma, 
                           nrounds = 1, 
                           watchlist = list(train = dtrainSigma),
                           verbose = verbose, 
                           base_score = ifelse(m==1, sigma_init, 0)
      )
      
      # predict output each round
      f_mu.predict <<- predict(f_mu, dtrainMu, outputmargin = TRUE)
      f_sigma.predict <<- predict(f_sigma, dtrainSigma, outputmargin = TRUE)
      
      train_score =  LogScore(y = label, 
                              mu = f_mu.predict, 
                              sigma = exp(f_sigma.predict)+1e-10) %>% mean()
      
      # set predict value as initial value next round
      setinfo(dtrainMu, "base_margin", predict(f_mu, dtrainMu, outputmargin = TRUE))
      setinfo(dtrainSigma, "base_margin", predict(f_sigma, dtrainSigma, outputmargin = TRUE))
      muModelList[[m]] = f_mu
      sigmaModelList[[m]] = f_sigma
      
      # early stopping
      
      if(best_eval > f_mu$evaluation_log$train_rmse  & best_log_eval > train_score) {
        best_eval = f_mu$evaluation_log$train_rmse
        best_log_eval = train_score
        best_iter = m
        cat("Iter:", best_iter,"train_-logP =",train_score, "\n")
      }else if((m - best_iter) >= early_stopping_rounds){
        m_stop = best_iter
        cat("Iter:", best_iter,"train_-logP =",train_score, "early_stopping: ", m, "->", best_iter, "rounds \n")
        break
      }
      
      mu.list = cbind(mu.list, f_mu.predict)
      sigma.list = cbind(sigma.list, f_sigma.predict)
    }
    cat("Model stop at ", best_iter, "round \n")
    mu.list = mu.list[,1 : best_iter]
    sigma.list = sigma.list[,1 : best_iter]
    return(list(muModelList=muModelList,
                sigmaModelList =sigmaModelList,
                mu.list = mu.list,
                sigma.list=sigma.list,
                m_stop = best_iter))
  }
}


getp = function(alpha, beta, mu, delta, y){
  p = alpha * sqrt((delta^2)+(y-mu)^2)
  return(p)
}

NigLogScore = function(alpha, beta, mu, delta, y){
  alpha = scale(alpha, t=0.001)
  delta = scale(delta, t=0.001)
  gamma = sqrt((alpha^2-beta^2) %>% sigmoid())
  p  = getp(alpha, beta, mu, delta, y)
  score = -log(alpha)-log(delta)-log(besselK(p, nu = 1)) + log(sqrt((y-mu)^2+delta^2)) -
    delta*gamma - beta * (y - mu)
  return(score)
}
sigmoid = function(x){
  value = 1/(1+exp(-x))
  return(value)
}
scale = function(x, t=0){
  x[x<t]=t
  return(x)
}
