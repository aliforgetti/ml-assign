Homework 1
================
Yaqoob, Ali
January 22, 2020

## Libraries

``` r
library('ElemStatLearn')
```

    ## Warning: package 'ElemStatLearn' was built under R version 3.6.2

## Loading Data

``` r
## load prostate data
data("prostate")
```

## Subset Training Data

``` r
## subset to training examples
prostate_train <- subset(prostate, train=TRUE)
```

## plot lcavol vs lpsa

``` r
plot_psa_data <- function(dat=prostate_train) {
  plot(dat$lpsa, dat$lcavol,
       xlab="log Prostate Screening Antigen (psa)",
       ylab="log Cancer Volume (lcavol)")
}
plot_psa_data()
```

![](homework_01_files/figure-gfm/unnamed-chunk-4-1.png)<!-- -->

## Loss functions defined

``` r
## L2 loss function
L2 <- function(y, yhat)
  (y-yhat)^2

## L1 loss function
L1 <- function(y, yhat)
  abs(y-yhat)
```

## fit simple linear model using numerical optimization

``` r
fit_lin <- function(y, x, loss, beta_init = c(-0.51, 0.75), beta_init_2 = c(-1.0, 0.0, -0.3), tau=0.5) {
  
  if(loss == 'L1'){
    loss <- get(loss)
    err <- function(beta)
      quantile(loss(y,  beta[1] + beta[2]*x), tau)
    beta <- optim(par = beta_init, fn = err) 
    # optimizes the beta value by minimizes the 
    return(beta)
  }
  
  if(loss == 'L2'){
    loss <- get(loss)
    err <- function(beta)
      mean(loss(y,  beta[1] + beta[2]*x))
    beta <- optim(par = beta_init, fn = err) 
    # optimizes the beta value by minimizes the 
    return(beta)
  }
  
  if(loss == 'L1_N'){
    loss <- L1
    err <- function(beta)
      quantile(loss(y,  beta[1] + beta[2]*exp(-beta[3]*x)), tau)
    beta <- optim(par = beta_init_2, fn = err) 
    # optimizes the beta value by minimizes the 
    return(beta)
  }
  
  if(loss == 'L2_N'){
    loss <- L2
    err <- function(beta)
      mean(loss(y,  beta[1] + beta[2]*exp(-beta[3]*x)))
    beta <- optim(par = beta_init_2, fn = err) 
    # optimizes the beta value by minimizes the 
    return(beta)
  }
  
}
```

## make predictions from linear model

``` r
predict_lin <- function(x, beta)
  beta[1] + beta[2]*x

predict_lin_2 <- function(x, beta)
  beta[1] + beta[2]*exp(-beta[3]*x)
```

## fit L1 linear model

``` r
lin_beta_L1 <- fit_lin(y=prostate_train$lcavol,
                    x=prostate_train$lpsa,
                    loss='L1')

lin_beta_L2 <- fit_lin(y=prostate_train$lcavol,
                    x=prostate_train$lpsa,
                    loss='L2')

lin_beta_L1_1 <- fit_lin(y=prostate_train$lcavol,
                    x=prostate_train$lpsa,
                    loss='L1', tau = 0.25)

lin_beta_L1_2 <- fit_lin(y=prostate_train$lcavol,
                    x=prostate_train$lpsa,
                    loss='L1', tau = 0.75)

lin_beta_L1_N <- fit_lin(y=prostate_train$lcavol,
                    x=prostate_train$lpsa,
                    loss='L1_N')

lin_beta_L2_N <- fit_lin(y=prostate_train$lcavol,
                    x=prostate_train$lpsa,
                    loss='L2_N')

lin_beta_L1_N1 <- fit_lin(y=prostate_train$lcavol,
                    x=prostate_train$lpsa,
                    loss='L1_N', tau = 0.25)

lin_beta_L1_N2 <- fit_lin(y=prostate_train$lcavol,
                    x=prostate_train$lpsa,
                    loss='L1_N', tau = 0.75)
```

## compute predictions for a grid of inputs

``` r
x_grid <- seq(min(prostate_train$lpsa),
              max(prostate_train$lpsa),
              length.out=100)
#Linear
lin_pred_1 <- predict_lin(x=x_grid, beta=lin_beta_L1$par)
lin_pred_2 <- predict_lin(x=x_grid, beta=lin_beta_L2$par)
lin_pred_3 <- predict_lin(x=x_grid, beta=lin_beta_L1_1$par)
lin_pred_4 <- predict_lin(x=x_grid, beta=lin_beta_L1_2$par)

#Non Linear
lin_pred_5 <- predict_lin_2(x=x_grid, beta=lin_beta_L1_N$par)
lin_pred_6 <- predict_lin_2(x=x_grid, beta=lin_beta_L2_N$par)
lin_pred_7 <- predict_lin_2(x=x_grid, beta=lin_beta_L1_N1$par)
lin_pred_8 <- predict_lin_2(x=x_grid, beta=lin_beta_L1_N2$par)
```

## Plotting the data

``` r
par(mfrow=c(1,2))

## plot data
plot_psa_data()
## plot predictions
lines(x=x_grid, y=lin_pred_1, col ='blue')
lines(x=x_grid, y=lin_pred_2, col ='red')
lines(x=x_grid, y=lin_pred_3, col ='grey')
lines(x=x_grid, y=lin_pred_4, col ='black')
title('Linear Model')
legend(x=3.75,y =0,
       legend = c("L1", "L2","tau = 25","tau = 75"),
       col=c("blue","red","grey","black"),
       pch = c(17,19),
       bty = "n",
       pt.cex = 0.75,
       cex = 0.75,
       text.col = "black",
       horiz = F ,
       inset = c(0.1, 0.1))

## plot data
plot_psa_data()
## plot predictions
lines(x=x_grid, y=lin_pred_5, col ='blue')
lines(x=x_grid, y=lin_pred_6, col ='red')
lines(x=x_grid, y=lin_pred_7, col ='grey')
lines(x=x_grid, y=lin_pred_8, col ='black')
title('Non-Linear Model')
legend(x=3.75,y =0,
       legend = c("L1", "L2","tau = 25","tau = 75"),
       col=c("blue","red","grey","black"),
       pch = c(17,19),
       bty = "n",
       pt.cex = 0.75,
       cex = 0.75,
       text.col = "black",
       horiz = F ,
       inset = c(0.1, 0.1))
```

![](homework_01_files/figure-gfm/unnamed-chunk-10-1.png)<!-- --> \`\`\`