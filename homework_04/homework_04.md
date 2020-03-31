Homework 4
================
Yaqoob, Ali
March 31, 2020

``` r
library(ElemStatLearn)
library(randomForest)
library(tidyverse)
library(caret)
```

``` r
data(vowel.train)
data(vowel.test)
```

``` r
head(vowel.train)
```

    ##   y    x.1   x.2    x.3   x.4    x.5   x.6    x.7    x.8    x.9   x.10
    ## 1 1 -3.639 0.418 -0.670 1.779 -0.168 1.627 -0.388  0.529 -0.874 -0.814
    ## 2 2 -3.327 0.496 -0.694 1.365 -0.265 1.933 -0.363  0.510 -0.621 -0.488
    ## 3 3 -2.120 0.894 -1.576 0.147 -0.707 1.559 -0.579  0.676 -0.809 -0.049
    ## 4 4 -2.287 1.809 -1.498 1.012 -1.053 1.060 -0.567  0.235 -0.091 -0.795
    ## 5 5 -2.598 1.938 -0.846 1.062 -1.633 0.764  0.394 -0.150  0.277 -0.396
    ## 6 6 -2.852 1.914 -0.755 0.825 -1.588 0.855  0.217 -0.246  0.238 -0.365

``` r
head(vowel.test)
```

    ##   y    x.1    x.2    x.3    x.4    x.5   x.6    x.7   x.8    x.9   x.10
    ## 1 1 -1.149 -0.904 -1.988  0.739 -0.060 1.206  0.864 1.196 -0.300 -0.467
    ## 2 2 -2.613 -0.092 -0.540  0.484  0.389 1.741  0.198 0.257 -0.375 -0.604
    ## 3 3 -2.505  0.632 -0.593  0.304  0.496 0.824 -0.162 0.181 -0.363 -0.764
    ## 4 4 -1.768  1.769 -1.142 -0.739 -0.086 0.120 -0.230 0.217 -0.009 -0.279
    ## 5 5 -2.671  3.155 -0.514  0.133 -0.964 0.234 -0.071 1.192  0.254 -0.471
    ## 6 6 -2.509  1.326  0.354  0.663 -0.724 0.418 -0.496 0.713  0.638 -0.204

1.  Convert the response variable in the “vowel.train” data frame to a
    factor variable prior to training, so that “randomForest” does
    classification rather than regression.

<!-- end list -->

``` r
vowel.train$y <- as.factor(vowel.train$y)
vowel.test$y <- as.factor(vowel.test$y) 
```

2.  Review the documentation for the “randomForest” function.
3.  Fit the random forest model to the vowel data using all of the 11
    features using the default values of the tuning parameters.

<!-- end list -->

``` r
vowel_rf <- randomForest(y ~., data = vowel.train)
```

``` r
plot(vowel_rf)
```

![](homework_04_files/figure-gfm/plot%20of%20the%20error-1.png)<!-- -->
4. Use 5-fold CV and tune the model by performing a grid search for the
following tuning parameters: 1) the number of variables randomly sampled
as candidates at each split; consider values 3, 4, and 5, and 2) the
minimum size of terminal nodes; consider a sequence (1, 5, 10, 20, 40,
and 80).

``` r
#misclassification function
error <- function(true, pred) {
  mean(true != pred, na.rm = T)
}

#tuning values
mtry <- c(3,4,5)
nodesize <- c(1,5,10, 40, 80)

params <- expand.grid(mtry = mtry,nodesize = nodesize)

n_folds <- 5

# create folds, stratified on the output, y
folds_i <- createFolds(factor(vowel.train$y), k = n_folds, list = F)

cv_ms_error <- matrix(nrow = n_folds, ncol = nrow(params))


for(j in 1:nrow(params)){
  
  for(i in 1:n_folds){
    train_ind <- which(folds_i != i)
    test_ind <- which(folds_i == i)
    train_k <- vowel.train[train_ind,]
    test_k <- vowel.test[test_ind,]
    
    rf_tune <- randomForest(y ~., 
                            data = train_k, 
                            xtest = na.omit(test_k[-1]), 
                            ytest = na.omit(test_k$y),
                            mtry = params[j,][[1]],
                            nodesize = params[j,][[2]])
    
    cv_ms_error[i,j] <- error(as.vector(na.omit(test_k$y)), rf_tune$test$predicted)
    
  }
}

cv_ms_error
```

    ##           [,1]      [,2]      [,3]      [,4]      [,5]      [,6]      [,7]
    ## [1,] 0.3626374 0.4065934 0.3846154 0.3626374 0.3956044 0.4065934 0.3846154
    ## [2,] 0.4468085 0.4574468 0.4893617 0.4680851 0.4787234 0.5106383 0.4787234
    ## [3,] 0.4591837 0.4795918 0.5102041 0.4591837 0.5102041 0.5204082 0.5000000
    ## [4,] 0.4157303 0.4382022 0.4382022 0.4269663 0.4157303 0.4494382 0.4044944
    ## [5,] 0.3777778 0.3666667 0.4111111 0.3888889 0.3888889 0.3888889 0.4111111
    ##           [,8]      [,9]     [,10]     [,11]     [,12]     [,13]     [,14]
    ## [1,] 0.4395604 0.4175824 0.5164835 0.5054945 0.4835165 0.5824176 0.6043956
    ## [2,] 0.4680851 0.4893617 0.5106383 0.4680851 0.5531915 0.5851064 0.5425532
    ## [3,] 0.5000000 0.5204082 0.5510204 0.5408163 0.5714286 0.6428571 0.6326531
    ## [4,] 0.4382022 0.4382022 0.5168539 0.5056180 0.5168539 0.5955056 0.6292135
    ## [5,] 0.4444444 0.4555556 0.4888889 0.4888889 0.4555556 0.5333333 0.5666667
    ##          [,15]
    ## [1,] 0.5934066
    ## [2,] 0.5744681
    ## [3,] 0.6632653
    ## [4,] 0.6292135
    ## [5,] 0.5666667

``` r
mean_cv <- colMeans(cv_ms_error)
mean_cv
```

    ##  [1] 0.4124275 0.4297002 0.4466989 0.4211523 0.4378302 0.4551934 0.4357889
    ##  [8] 0.4580584 0.4642220 0.5167770 0.5017806 0.5161092 0.5878440 0.5950964
    ## [15] 0.6054040

5.  With the tuned model, make predictions using the majority vote
    method, and compute the misclassification rate using the
    ‘vowel.test’ data.

<!-- end list -->

``` r
best_index <- which(mean_cv == min(mean_cv))[1]
params[best_index,]
```

    ##   mtry nodesize
    ## 1    3        1

These are the best tuning parameters with the lowest misclassification
rate.

``` r
# final model tunned
rf_tune <- randomForest(y ~., 
                        data = train_k, 
                        xtest = na.omit(vowel.test[-1]), 
                        ytest = na.omit(vowel.test$y),
                        mtry = params[best_index,][[1]],
                        nodesize = params[best_index,][[2]])
```

``` r
# true value
y <- as.double(as.vector(vowel.test$y))
# predicted value from the model
y_hat <- as.double(as.vector(rf_tune$test$predicted))

error(y,y_hat) # misclassification rate of the final model tuned with the best parameters
```

    ## [1] 0.4155844
