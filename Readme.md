# XGBoost-Distribution-Estimation-Model



依據XGBoost的框架實作一個可以預測樣本分配估計的模型，以更好的捕捉資料的不確定性。

## Implementation
將XGBoost的迴歸樹 Loss function 更改為最大概似法估計(Maximum Likelihood Estimation)進行梯度下降如下所示。
將$L(y)$變更為$-logP(y)$。 其中$P(y)$以常態分配(Normal distribution)之p.d.f為例。
$$
L(y) = \frac{1}{2}(y_i-\hat{y_i})^2 \rightarrow -log P(y)
$$

$$
\begin{split}
\large
P(y) &= \large\frac{1}{\sigma\sqrt{2\pi}}\large e^{-\frac{1}{2}(\frac{y-\mu}{\sigma})^2}
\end{split}
$$

參考Ngboost(2019)所使用之Fisher information metric，以正確計算每個參數之梯度。
$$
\begin{split}
\text{Set t} &= log(\sigma) \\
\\
l(\theta) &=-logP(y)\\
&= \frac{1}{2} log(\sigma^2)+\frac{(y-\mu)^2}{2\sigma^2}\\
&= t +\frac{(y-\mu)^2}{2e^{2t}}
\end{split}\\
$$
$$
\begin{split}
&I_{\theta} =E_{y\sim P_{\theta}}
\large\left[
\begin{array}{cc}
\frac{\partial^2 l}{\partial^2 \mu}  
&
\frac{\partial^2 l}{\partial \mu\partial t} 
\\
\frac{\partial^2 l}{\partial t\partial\mu}
& \frac{\partial^2 l}{\partial^2 t}\\
\end{array}
\right] = E_{y\sim P_{\theta}}
\large\left[
\begin{array}{cc}
 \frac{1}{e^{2t}}  
&
\frac{2(y-\mu)}{e^{2t}} 
\\
\frac{2(y-\mu)}{e^{2t}} 
&  \frac{2(y-\mu)^2}{e^{2t}} \\
\end{array}
\right]
\end{split}
$$

經整理後設計兩個不同的XGBoost Decision Tree 模型$\large f_{\mu}(x)、 f_{t}(x)$以分別預測常態分佈的兩個參數，其中之$\large g、h$值如以下所示：

$$
\begin{split}
\\
f_{\mu} 
\left 
\{
\begin{array}{cc}
g_i  &=  -\frac{y-\mu}{1+e^{2t} \epsilon}& \\
h_i  &= 1 
\end{array}
\right. 
\end{split}
$$

$$
\begin{split}
f_{\ t} 
\left 
\{
\begin{array}{cc}
g_i &=\frac{1}{2}-\frac{(y-\mu)^2}{2e^{2t}} \\
h_i &= 1\\
\end{array}
\right. 
\end{split}
$$



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



詳細使用方法可以參考`walkthrough.R`
