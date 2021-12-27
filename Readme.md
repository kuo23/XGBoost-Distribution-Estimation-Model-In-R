# XGBoost-Distribution-Estimation-Model



依據XGBoost的框架實作一個可以預測樣本分配估計的模型，以更好的捕捉資料的不確定性。

## Implementation
將XGBoost的迴歸樹 Loss function 更改為最大概似法估計(Maximum Likelihood Estimation)進行梯度下降如下所示。
將$L(y)$變更為$-logP(y)$。 其中$P(y)$以常態分配(Normal distribution)之p.d.f為例。
<p align="center">
<img src="https://render.githubusercontent.com/render/math?math=L(y)%20%3D%20%5Cfrac%7B1%7D%7B2%7D(y_i-%5Chat%7By_i%7D)%5E2%20%5Crightarrow%20-log%20P(y)">
</p>
<p align="center">
<img src="https://render.githubusercontent.com/render/math?math=%5Cbegin%7Bsplit%7D%0A%5Clarge%0AP(y)%20%26%3D%20%5Clarge%5Cfrac%7B1%7D%7B%5Csigma%5Csqrt%7B2%5Cpi%7D%7D%5Clarge%20e%5E%7B-%5Cfrac%7B1%7D%7B2%7D(%5Cfrac%7By-%5Cmu%7D%7B%5Csigma%7D)%5E2%7D%0A%5Cend%7Bsplit%7D"></p>

參考[Ngboost(2019)](https://github.com/stanfordmlgroup/ngboost)所使用之Fisher information metric，以正確計算每個參數之梯度。
<p align="center">
<img src="https://render.githubusercontent.com/render/math?math=%5Cbegin%7Bsplit%7D%0A%5Ctext%7BSet%20t%7D%20%26%3D%20log(%5Csigma)%20%5C%5C%0A%5C%5C%0Al(%5Ctheta)%20%26%3D-logP(y)%5C%5C%0A%26%3D%20%5Cfrac%7B1%7D%7B2%7D%20log(%5Csigma%5E2)%2B%5Cfrac%7B(y-%5Cmu)%5E2%7D%7B2%5Csigma%5E2%7D%5C%5C%0A%26%3D%20t%20%2B%5Cfrac%7B(y-%5Cmu)%5E2%7D%7B2e%5E%7B2t%7D%7D%0A%5Cend%7Bsplit%7D%5C%5C">
</p>
<p align="center">
<img src="https://render.githubusercontent.com/render/math?math=%5Cbegin%7Bsplit%7D%0A%26I_%7B%5Ctheta%7D%20%3DE_%7By%5Csim%20P_%7B%5Ctheta%7D%7D%0A%5Clarge%5Cleft%5B%0A%5Cbegin%7Barray%7D%7Bcc%7D%0A%5Cfrac%7B%5Cpartial%5E2%20l%7D%7B%5Cpartial%5E2%20%5Cmu%7D%20%20%0A%26%0A%5Cfrac%7B%5Cpartial%5E2%20l%7D%7B%5Cpartial%20%5Cmu%5Cpartial%20t%7D%20%0A%5C%5C%0A%5Cfrac%7B%5Cpartial%5E2%20l%7D%7B%5Cpartial%20t%5Cpartial%5Cmu%7D%0A%26%20%5Cfrac%7B%5Cpartial%5E2%20l%7D%7B%5Cpartial%5E2%20t%7D%5C%5C%0A%5Cend%7Barray%7D%0A%5Cright%5D%20%3D%20E_%7By%5Csim%20P_%7B%5Ctheta%7D%7D%0A%5Clarge%5Cleft%5B%0A%5Cbegin%7Barray%7D%7Bcc%7D%0A%20%5Cfrac%7B1%7D%7Be%5E%7B2t%7D%7D%20%20%0A%26%0A%5Cfrac%7B2(y-%5Cmu)%7D%7Be%5E%7B2t%7D%7D%20%0A%5C%5C%0A%5Cfrac%7B2(y-%5Cmu)%7D%7Be%5E%7B2t%7D%7D%20%0A%26%20%20%5Cfrac%7B2(y-%5Cmu)%5E2%7D%7Be%5E%7B2t%7D%7D%20%5C%5C%0A%5Cend%7Barray%7D%0A%5Cright%5D%0A%5Cend%7Bsplit%7D">
</p>
經整理後設計兩個不同的XGBoost Decision Tree 模型$\large f_{\mu}(x)、 f_{t}(x)$以分別預測常態分佈的兩個參數，其中之$\large g、h$值如以下所示：
<p align="center">
<img src="https://render.githubusercontent.com/render/math?math=%5Cbegin%7Bsplit%7D%0A%5C%5C%0Af_%7B%5Cmu%7D%20%0A%5Cleft%20%0A%5C%7B%0A%5Cbegin%7Barray%7D%7Bcc%7D%0Ag_i%20%20%26%3D%20%20-%5Cfrac%7By-%5Cmu%7D%7B1%2Be%5E%7B2t%7D%20%5Cepsilon%7D%26%20%5C%5C%0Ah_i%20%20%26%3D%201%20%0A%5Cend%7Barray%7D%0A%5Cright.%20%0A%5Cend%7Bsplit%7D">
</p>
<p align="center">
<img src="https://render.githubusercontent.com/render/math?math=%5Cbegin%7Bsplit%7D%0Af_%7B%5C%20t%7D%20%0A%5Cleft%20%0A%5C%7B%0A%5Cbegin%7Barray%7D%7Bcc%7D%0Ag_i%20%26%3D%5Cfrac%7B1%7D%7B2%7D-%5Cfrac%7B(y-%5Cmu)%5E2%7D%7B2e%5E%7B2t%7D%7D%20%5C%5C%0Ah_i%20%26%3D%201%5C%5C%0A%5Cend%7Barray%7D%0A%5Cright.%20%0A%5Cend%7Bsplit%7D">
</p>


## Verification

- 收斂情形

  可以針對估計不同的樣本得到不同的估計區間。

  - ![](p3.gif)

## Installation

```R
install.packages("xgboost")
```



## Usage

```R
    MuLoss <- function(preds, dtrain, sig = f_sigma.predict) {
      labels <- getinfo(dtrain, "label")
      preds <- preds
      grad <- -((labels - preds)/(1+(exp(2*sig)*0.00001)))
      hess <- (labels - labels + 1)
      return(list(grad = grad, hess = hess))
    }


    SigmaLoss <- function(preds, dtrain, mu = f_mu.predict) {
      labels <- getinfo(dtrain, "label")
      preds <- preds
      grad <- -((((labels - mu)^2)/(2 * exp(2 * preds))) - 0.5)
      hess <-(labels - labels + 1)
      return(list(grad = grad, hess = hess))
    }

model = CustomXgb(trainData = trainData,
                    col = col,
                    iter_rounds = iter_rounds,
                    label =  trainData$y,
                    valData = valData, 
                    valLabel = valData$y)
```



詳細使用方法可以參考`src/walkthrough.R`
