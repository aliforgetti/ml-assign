Homework 3
================
Yaqoob, Ali
March 15, 2020

``` r
library('MASS') ## for 'mcycle'
library('manipulate') ## for 'manipulate'
```

1.  Randomly split the mcycle data into training (75%) and validation
    (25%) subsets.

<!-- end list -->

``` r
data <- mcycle
sample <- sample.int(n = nrow(data), size = floor(.75*nrow(data)), replace = F)
train <- data[sample, ]
test  <- data[-sample, ]
```

``` r
yt <- train$accel
xt <- matrix(train$times, length(train$times), 1)
yv <- test$accel
xv <-matrix(test$times, length(test$times), 1)
```

``` r
plot(xt, yt, xlab="Time (ms)", ylab="Acceleration (g)")
```

![](homework_03_files/figure-gfm/unnamed-chunk-4-1.png)<!-- -->

2.  Using the mcycle data, consider predicting the mean acceleration as
    a function of time. Use the Nadaraya-Watson method with the k-NN
    kernel function to create a series of prediction models by varying
    the tuning parameter over a sequence of values.

<!-- end list -->

``` r
## k-NN kernel function
## x  - n x p matrix of training inputs
## x0 - 1 x p input where to make prediction
## k  - number of nearest neighbors
kernel_k_nearest_neighbors <- function(x, x0, k=1) {
  ## compute distance betwen each x and x0
  z <- t(t(x) - x0)
  d <- sqrt(rowSums(z*z))

  ## initialize kernel weights to zero
  w <- rep(0, length(d))
  
  ## set weight to 1 for k nearest neighbors
  w[order(d)[1:k]] <- 1
  
  return(w)
}
```

``` r
## Make predictions using the NW method
## y  - n x 1 vector of training outputs
## x  - n x p matrix of training inputs
## x0 - m x p matrix where to make predictions
## kern  - kernel function to use
## ... - arguments to pass to kernel function
nadaraya_watson <- function(y, x, x0, kern, ...) {
  k <- t(apply(x0, 1, function(x0_) {
    k_ <- kern(x, x0_, ...)
    k_/sum(k_)
  }))
  yhat <- drop(k %*% y)
  attr(yhat, 'k') <- k
  return(yhat)
}
```

Create models for different values of k

``` r
## Create models using NW method and KNN Kernel
## y  - n x 1 vector of training outputs
## x  - n x p matrix of training inputs
## x0 - m x p matrix where to make predictions
## kstart  - starting value of tuning parameter
## kend  - end value of tuning parameter
## ksize - size of the vector to tune over

nwk_models <- function(y, x, x0, kstart = 1, kend = 98, ksize = 30){
  k <- seq(kstart,kend,length.out = ksize)
  k <- floor(k)
  pred_mat <- matrix( nrow = length(k), ncol = length(x0))
  
  x_plot <- matrix(seq(min(x),max(x),length.out=100),100,1)
  
  for(i in 1:length(k)){
    pred_mat[i,] <- nadaraya_watson(y, x, x0,
                                    kern = kernel_k_nearest_neighbors, k = k[i])
  }
  
  pred_mat
}
```

3.  With the squared-error loss function, compute and plot the training
    error, AIC, BIC, and validation error (using the validation data) as
    functions of the tuning parameter.

Compute and plot the training error as functions of the tuning
parameter.

``` r
## loss function
## y    - train/test y
## yhat - predictions at train/test x
loss_squared_error <- function(y, yhat)
  (y - yhat)^2
```

``` r
## test/train error
## y    - train/test y
## yhat - predictions at train/test x
## loss - loss function
error <- function(y, yhat, loss=loss_squared_error)
  mean(loss(y, yhat))
```

``` r
ksize = 30
train_pred <- nwk_models(yt, xt, xt, ksize = ksize)
training_error <- rep(NA, nrow(train_pred))

for(i in 1:length(training_error)){
  training_error[i] <- error(yt, train_pred[i,])
}
kvals = floor(seq(1,98,length.out = ksize))
plot(x =kvals,y=training_error, ylab = "Training Error", xlab = "K Values",type = 'o')
```

![](homework_03_files/figure-gfm/unnamed-chunk-10-1.png)<!-- -->

Compute and plot the AIC and BIC as a function of the tuning parameter.

``` r
## Compute effective df using NW method
## y  - n x 1 vector of training outputs
## x  - n x p matrix of training inputs
## kern  - kernel function to use
## ... - arguments to pass to kernel function
effective_df <- function(y, x, kern, ...) {
  y_hat <- nadaraya_watson(y, x, x,
    kern=kern, ...)
  sum(diag(attr(y_hat, 'k')))
}

## AIC
## y    - training y
## yhat - predictions at training x
## d    - effective degrees of freedom
aic <- function(y, yhat, d)
  error(y, yhat) + 2/length(y)*d

## BIC
## y    - training y
## yhat - predictions at training x
## d    - effective degrees of freedom
bic <- function(y, yhat, d)
  error(y, yhat) + (log(length(y))/length(y))*d

aic_vals <- rep(NA, nrow(train_pred))
bic_vals <- rep(NA, nrow(train_pred))
for(i in 1:length(aic_vals)){
  ## compute effective degrees of freedom
  edf <- effective_df(yt, xt, kern = kernel_k_nearest_neighbors, k = kvals[i])
  #print(edf)
  aic_vals[i] <- aic(yt, train_pred[i,], edf)
  bic_vals[i] <- bic(yt, train_pred[i,], edf)
}

par(mfrow = c(1,2))
plot(x =kvals, y=bic_vals, ylab = "BIC", xlab = "K Values", type = 'o')
plot(x =kvals, y=aic_vals, ylab = "AIC", xlab = "K Values", type = 'o')
```

![](homework_03_files/figure-gfm/Plotting%20AIC%20and%20BIC%20against%20K%20values-1.png)<!-- -->

Compute and plot validation error as function of tuning parameter.

``` r
valid_pred <- nwk_models(yt,xt,xv,ksize = 30)
validation_error <- rep(NA, nrow(valid_pred))

for(i in 1:length(training_error)){
  validation_error[i] <- error(yv, valid_pred[i,])
}
kvals = floor(seq(1,98,length.out = ksize))
par(xpd = T)
plot(x =kvals,y=training_error, ylab = "Error", xlab = "K Values",type = 'l', ylim = c(0,4000), col = "red")
lines(x = kvals, y = validation_error, col = "blue", lty = 2)
legend(1, 3900, legend=c("Training Error", "Validation Error"),
       col=c("red", "blue"), lty=1:2, cex=0.8)
```

![](homework_03_files/figure-gfm/unnamed-chunk-11-1.png)<!-- -->

4.  For each value of the tuning parameter, Perform 5-fold
    cross-validation using the combined training and validation data.
    This results in 5 estimates of test error per tuning parameter
    value.

<!-- end list -->

``` r
n_train <- nrow(mcycle)
n_folds <- 5
folds_i <- sample(rep(1:n_folds, length.out = n_train))
cv_test_error <- matrix(nrow = n_folds, ncol = length(kvals))

for(j in 1:length(kvals) ){
  for(i in 1:n_folds){
    train_ind <- which(folds_i != i)
    test_ind <- which(folds_i == i)
    train_k <- mcycle[train_ind,]
    test_k <- mcycle[test_ind,]
    yt <- train_k$accel
    xt <- matrix(train_k$times, length(train_k$times), 1)
    yv <- test_k$accel
    xv <-matrix(test_k$times, length(test_k$times), 1)
    yv_hat <- nadaraya_watson(yt, xt, xv,
                             kern = kernel_k_nearest_neighbors, k = kvals[j])
    cv_test_error[i,j] <-  error(yv, yv_hat)

  }
}


cv_test_error <- as.data.frame(cv_test_error)
colnames(cv_test_error) <- kvals
cv_test_error
```

    ##           1        4        7       11       14       17       21       24
    ## 1  803.1156 523.5363 569.8149 470.8881 562.0558 559.1053 657.2050 738.8274
    ## 2  849.1741 770.7213 673.9367 697.3683 633.7176 649.9420 734.3925 832.1765
    ## 3  969.9641 459.7056 494.2277 446.6388 434.6580 456.8694 498.4412 590.0693
    ## 4 1379.3735 882.1139 699.5627 659.5973 692.0333 724.7567 775.2097 868.4014
    ## 5 1281.4677 472.8133 560.6511 533.7967 621.4078 691.7901 737.1198 853.9176
    ##         27        31        34        37        41       44       47       51
    ## 1 779.8068  854.3959  913.2374  944.4713 1002.1711 1033.314 1105.200 1252.086
    ## 2 949.4763 1092.1996 1127.3346 1233.7232 1341.9181 1465.284 1598.362 1805.113
    ## 3 642.1166  775.5359  844.0437  882.6631  927.3627 1005.214 1084.735 1226.322
    ## 4 982.8068 1078.4746 1139.5254 1263.0006 1367.0896 1498.219 1622.668 1910.819
    ## 5 928.1155 1121.6521 1279.7954 1391.9695 1628.4401 1771.598 1985.714 2232.742
    ##         54       57       61       64       67       71       74       77
    ## 1 1349.773 1429.310 1538.317 1612.424 1687.032 1756.555 1750.857 1777.291
    ## 2 1915.777 2014.016 2127.419 2170.402 2170.436 2165.185 2173.736 2190.147
    ## 3 1333.223 1425.766 1579.997 1679.583 1756.331 1837.440 1882.459 1905.584
    ## 4 2095.432 2297.766 2547.995 2695.167 2806.418 2915.405 2964.589 2961.728
    ## 5 2470.297 2628.587 2791.696 2919.530 2972.367 3000.669 2991.929 2943.936
    ##         81       84       87       91       94       98
    ## 1 1812.420 1838.294 1856.410 1878.567 1893.825 1907.055
    ## 2 2189.306 2192.669 2191.882 2198.976 2205.182 2201.718
    ## 3 1960.204 1977.353 1998.872 2024.345 2036.152 2058.182
    ## 4 2973.065 2971.214 2960.267 2945.017 2926.215 2913.359
    ## 5 2921.419 2875.751 2861.524 2822.206 2794.410 2749.355

Each column in cv\_test\_error represents one k value (tuning parameter)
and the five CV test errors associated with each k value.

5.  Plot the CV-estimated test error (average of the five estimates from
    each fold) as a function of the tuning parameter. Add vertical line
    segments to the figure (using the segments function in R) that
    represent one “standard error” of the CV-estimated test error
    (standard deviation of the five estimates from each fold).

6.  Interpret the resulting figures and select a suitable value for the
    tuning parameter.

<!-- end list -->

``` r
cv_mean <- apply(cv_test_error, 2, mean)
cv_sd   <- apply(cv_test_error, 2, sd)


plot(x = kvals , y = cv_mean, type = 'o', ylab = "Average Test Error", xlab = "K Values", ylim = c(0,4000))
segments(x0=kvals, x1=kvals,
         y0=cv_mean - cv_sd,
         y1=cv_mean + cv_sd)
best_idx <- which.min(cv_mean)
points(x=best_idx, y=cv_mean[best_idx], pch=20)
abline(h=cv_mean[best_idx] + cv_sd[best_idx], lty=3)
```

![](homework_03_files/figure-gfm/unnamed-chunk-13-1.png)<!-- -->

As we increase the value of k, the model tends moves towards lower
variance.

``` r
optimal <- cv_mean[best_idx] + cv_sd[best_idx]
kvals[which(abs(cv_mean - optimal) == min(abs(cv_mean - optimal)))]
```

    ## [1] 21

This is the best tuning parameter value as it one standard deviation of
the lowest test error rate.
