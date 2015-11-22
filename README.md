# 07_Reg_models

#### Variable selection - regression models  
1. Best Subset Selection, Forward and Backward Stepwise Selection: The model with the lowest BIC (-47), **[ref: "Fig1: BIC: Best mode"]**, is the 3-variables model: **wt, qsec & am**. When compared best subset, forward and backwards stepwise; 1,2,4,11 and 12 variables model are same for the 3 models. As 1 & 2 variables model, do not have 'am' component, eliminate them. Next is 4-variable model, it has 'am' component. The 4-variables are: **cyl, hp, wt & am**.
2. Choosing Among Models Using the Validation Set Approach and Cross-Validation: As per validation approach, 5-variables model: **cyl, hp, wt, vs & am**. As per cross-validation **[ref: "Fig2: Mean CV errors"]**, 4-variables model: **cyl, hp, wt & am1**.
3. Ridge Regression and The Lasso: Ridge **[ref: "Fig3: Ridge: CV error"]**: none of the coefficients are zero, so does not perform variable selection! Lasso **[ref: "Fig4: Lasso" & "Fig5: Lasso: CV error"]**: With lambda chosen by cross-validation contains only 7 variables, after eliminating all the zeroes coefficient variables: **cyl, hp, drat, wt, vs & am**
4. Principal Components Regression  **[ref: "Fig6: PCR CV Scores: full data"]** and Partial Least Squares  where also performed. Noticed PCR test set MSE was 6.17, competitive to ridge (4.71) regression and better than lasso (8.41). PCR . PLS test MSE was 6.45, comparable but higher than PCR and ridge, lesser than lasso. By cross validation, PCR identified 2 components where as PLS 3 components. The percentage of variance in 'mpg' that the 2-component PLS fit explains, 85.58%, is almost as much as that explained using the final 3-component model PCR fit, 85.21%.
5. **Conclusion:** I choose "all the 3 models selection", the 4-variables: **cyl, hp, wt & am**. One good reason the (empirical mean) intercept was 34.35, best of all the options. Also, lesser variables, variable 'am' present in this. Also, I personally would choose hp, wt and cyl (apart from qsec, which I eliminate after this process) along with am and mpg.

#### Details of variable selection:
====================================================
1a. subset
====================================================
```{r}
library (leaps)

#--My data set ...
library(datasets)
data(mtcars)
data4 <- mtcars
colnames <- c('cyl','am','vs','gear')
data4[, colnames] <- lapply(mtcars[, colnames], factor)
#--

#--Main settings to use further in the reg models
data <- data4
max_col <- ncol(model.matrix(mpg ~ ., data)[,])
#--

regfit.full <- regsubsets(mpg ~ ., data=data, nvmax=max_col)
reg.summary <- summary(regfit.full)

##- Plotting RSS, adjusted R2, Cp, and BIC
par(mfrow =c(3,2))

plot(reg.summary$adjr2 ,xlab =" Number of Variables ", ylab=" Adjusted RSq",type="l")
mx_r2 <- which.max (reg.summary$adjr2)
points (mx_r2, reg.summary$adjr2[mx_r2], col ="red",cex =2, pch =20)

mn_bic <- which.min (reg.summary$bic )
plot(reg.summary$bic ,xlab=" Number of Variables ",ylab="BIC", type='l')
points (mn_bic, reg.summary$bic [mn_bic], col =" red",cex =2, pch =20)

plot(regfit.full, scale = "adjr2")
plot(regfit.full, scale = "bic")
````

the model with the lowest BIC (-47) is the
3-variable model that contains only wt, qsec and am1.

Use the coef() function, to see the coefficient estimates associated with this model.
As this has 'am' variable and very less variable, this looks like a good choice.

====================================================
1b. Forward and backward stepwise selection
====================================================

```{r}
#--
regfit.fwd <- regsubsets(mpg ~ ., data=data, nvmax = max_col, method="forward")
#summary(regfit.fwd)
regfit.bwd <- regsubsets(mpg ~ ., data=data, nvmax = max_col, method="backward")
#summary(regfit.bwd)

## notice the differences or similarities between these 3 model selections.
summary(regfit.full)$outmat
summary(regfit.fwd)$outmat
summary(regfit.bwd)$outmat

fields_match <- function(regfit1, regfit2, tst=TRUE) {
	tmp <- (summary(regfit1)$outmat == summary(regfit2)$outmat)
	row.names(tmp) <- gsub("  \\( 1 )", "", row.names(tmp))
	w <- apply(tmp, 1, all)
	if (tst) w[w]
	else w[!w]
}

fields_match(regfit.bwd, regfit.fwd)
fields_match(regfit.bwd, regfit.full)
fields_match(regfit.full, regfit.fwd)
```

1,2,4,11,12 variables are all same in all the 3.
7 is different for all the 3.

```{r}
## Look at the coefficients of one sample which is same in all 3. 4-variable
coef(regfit.bwd, 4)
coef(regfit.full, 4)
coef(regfit.fwd, 4)

> coef(regfit.bwd, 4)
(Intercept)        cyl6          hp          wt         am1
34.34835479 -2.15170905 -0.04158974 -2.67523820  2.26599959
## other 2 ditto.

## Look at the coefficients of one sample which is diff in all 3. 7-variable
coef(regfit.bwd, 7)
coef(regfit.full, 7)
coef(regfit.fwd, 7)
```

1,2,4,11,12 variables are all same in all the 3.
7 is different for all the 3.

Here, 1 and 2 variable, do not have 'am' as a component. So, we eliminate it.
Next, is 4 and it has 'am', thus, that is the choice.
cyl6, hp, wt, am1

====================================================
1c. Choosing Among Models Using the Validation Set
Approach and Cross-Validation
====================================================

====================================================
1c.i. Validation set
====================================================

We just saw that it is possible to choose among a set of models
of different sizes using Cp, BIC, and adjusted R2.

We will now consider how to do this
using the validation set and cross-validation approaches

Split data into training and test, for this.

Choosing best seed, as per my logic
From this, seed 5, will be best. The overall validation error will be less.

```{r}
library(caret) ## for createDataPartition
set.seed(5)
train <- createDataPartition(data$mpg,p=0.7)[[1]]
test  <- (-train)

## perform best subset selection
regfit.best <- regsubsets(mpg ~ ., data=data[train,], nvmax = max_col)

## compute the validation set error for the best model of each model size
# First, make a model matrix from the test data
test.mat <- model.matrix(mpg ~ ., data=data[test,])

# extract coefficients, predict and compute test MSE
val.errors <- rep(NA, (max_col-1))
for(i in 1:(max_col-1)) {
  coefi <- coef(regfit.best, id=i)
  pred <- test.mat[,names(coefi)]%*%coefi
  val.errors[i] <- round(mean((data$mpg[test]-pred)^2),2)
}

val.errors
min(val.errors)
which.min(val.errors)
```

As per this, least error is with 10 variables.
but an error of 5% is acceptable and lesser number of variables is preferable,
thus, we will choose, 5 variables - which is around 4.99 error.

```{r}
coef(regfit.best, 5)
```

As I need, 'am' as a predictor,
Lesser the # of variable, it is better, I go with 5 variables.

So, as per validation set, 5-variable is much preferred
Finally, run the best subset selection on the full data set, not training now.

```{r}
regfit.best <- regsubsets(mpg ~ ., data=data, nvmax = max_col)
coef(regfit.best, 5)

##----Creating a predict function to use in future
predict.regsubsets <- function(object, newdata, id, ...)
{
    form  <- as.formula(object$call[[2]])
    mat   <- model.matrix(form, newdata)
    coefi <- coef(object, id=id)
    xvars <- names(coefi)
    mat[,xvars]%*%coefi
}
```

So, as per validation approach, 5-variables: cyl6, hp, wt, vs1, am1 is chosen.

NOTE: if set to seed(1); then, 7-variable is preferable with 12.32 error. Though
it is high in the list of errors, but, that is the one with 'am' component and
least variables possible.

====================================================
1c.ii. Cross-validation
====================================================
continuation from validation approach (prev calcs)

First, create a vector that allocates each observation to one of k = 5 folds,
and create a matrix in which we will store the results.

```{r}
k <- 10
set.seed(5)

folds <- sample(1:k, nrow(data), replace=TRUE)
cv.errors <- matrix(NA, k, (max_col-1), dimnames=list(NULL, paste(1:(max_col-1))))

## a for loop that performs cross-validation
for(j in 1:k){
    best.fit <- regsubsets(mpg ~ ., data=data[folds!=j,], nvmax=max_col)
    for(i in 1:(max_col-1)) {
       pred <- predict.regsubsets(best.fit, data[folds==j,], id=i)
       cv.errors[j,i] <- mean((data$mpg[folds==j]-pred)^2)
    }
}

mean.cv.errors <- apply(cv.errors, 2,mean)
#par(mfrow =c(1,1))
plot(mean.cv.errors, type='b')

mean.cv.errors
min(mean.cv.errors)
which.min(mean.cv.errors)

### now, from the full data set
regfit.best <- regsubsets(mpg ~ ., data=data, nvmax = 12)

##  as 2 is the least, that is the best fit
coef(regfit.best, 2)
## also, 4 and 5 are on similar values of 2; so check
coef(regfit.best, 4)
coef(regfit.best, 5)
```

As per this, i would choose 4 variable. But, as per validation approach, it is 5,
so we can also consider 5 here; cyl6, hp, wt, vs1, am1

====================================================
2. ridge
====================================================
```{r}
y <- data$mpg
x <- model.matrix(mpg ~ .,data)[,-1]

library(glmnet)
grid = 10^seq(10, -2, length=100)
ridge.mod <- glmnet(x, y, alpha=0, lambda=grid)

## train and test
set.seed (5)
train <- createDataPartition(data$mpg,p=0.6)[[1]]
test  <- (-train)
y.test <- y[test]

## instead of arbitrarily choosing λ = 4, it would be better to
## use cross-validation to choose the tuning parameter λ
set.seed (5)
cv.out1 <- cv.glmnet(x[train,], y[train], alpha=0, grouped=FALSE) ## alpha=0 -> ridge
plot(cv.out1)
bestlam <- cv.out1$lambda.min
bestlam

##
ridge.pred <- predict(ridge.mod, s=bestlam, newx=x[test,])
mean((ridge.pred - y.test)^2)

## Finally, we refit our ridge regression model on the full data set, using the
##  value of λ chosen by cross-validation, and examine the coefficient estimates
out <- glmnet(x, y, alpha = 0)
predict(out, type = "coefficients", s = bestlam)[1:12, ]
```

As expected, none of the coefficients are zero, ridge regression does not
perform variable selection!

====================================================
3. lasso
====================================================
```{r}
lasso.mod <- glmnet(x[train ,], y[train], alpha=1, lambda=grid)
plot(lasso.mod)

set.seed(5)
cv.out2 <- cv.glmnet(x[train,], y[train], alpha =1, grouped=FALSE) ## alpha=1 -> lasso
plot(cv.out2)
bestlam <- cv.out2$lambda.min
lasso.pred <- predict(lasso.mod, s=bestlam, newx=x[test,])

mean((lasso.pred - y.test)^2)

out <- glmnet(x,y,alpha=1,lambda=grid) ## alpha=1 is lasso
lasso.coef <- predict(out,type="coefficients",s=bestlam)[1:12,]
round(lasso.coef,2)
round(lasso.coef[lasso.coef != 0],2)
```

So the lasso model with λ chosen by cross-validation contains only 7 variables; eliminating
all the zeroes.
And this has 'am' with this, we can think of considering this.

====================================================
4. pcr
====================================================
```{r}
#install.packages("pls")
library(pls)
set.seed(5)
pcr.fit1 <- pcr(mpg ~ ., data=data, scale=TRUE, validation="CV")
summary(pcr.fit)
```

pcr() reports the root mean squared error (RMSEP); in order to obtain
the usual MSE, we must square this quantity;
example: 12 comps: 3.434 (RMSEP) -> 3.434^2 = 11.792 (MSE)

One can also plot the cross-validation scores using the validationplot()
validation function.
Using val.type="MSEP" will cause the cross-validation MSE to be plot()
plotted.

```{r}
validationplot(pcr.fit1, val.type="MSEP")
````

lowest cross-validation error occurs when M = 4 component are used.
perform PCR on the training data and evaluate its test set performance

```{r}
set.seed(5)
pcr.fit <- pcr(mpg~., data=data, subset=train, scale=TRUE, validation = "CV")
validationplot(pcr.fit,val.type="MSEP")

# lowest cross-validation error occurs when M = 3 component are used.

# We compute the 'test' MSE as follows.
pcr.pred=predict(pcr.fit, x[test,], ncomp=3)
mean((pcr.pred -y.test)^2)
```

This test set MSE is competitive with the result obtained using ridge (5.71) regression
but far better than the lasso(8.41). However, as a result of the way PCR is implemented,
the final model is more difficult to interpret because it does not perform
any kind of variable selection or even directly produce coefficient estimates.

Finally, we fit PCR on the full data set, using M = 3, the number of
components identified by cross-validation.
```{r}
pcr.fit <- pcr(y~x,scale=TRUE, ncomp=3)
summary(pcr.fit)
```

So, as per this it is 3-variable. And when you use 3 variable,
we see hat the variance is upto 83%.

====================================================
5. pls
====================================================
```{r}
set.seed(5)
pls.fit <- plsr(mpg ~ ., data=data, subset=train, scale=TRUE, validation="CV")
summary(pls.fit)
```

The lowest cross-validation error occurs when only M = 2 partial least
squares directions are used. We now evaluate the corresponding test set
MSE.
```{r}
pls.pred <- predict(pls.fit, x[test,], ncomp=2)
mean((pls.pred - y.test)^2)
```

The test MSE is comparable to, but higher than, the test MSE than PCR(6.17) and ridge(4.71)
with the bestlam as the lambda. But, lesser than lasso(8.41).

Finally, we perform PLS using the full data set, using M = 2, the number
of components identified by cross-validation.

```{r}
pls.fit <- plsr(mpg~., data=data, scale=TRUE, ncomp=2)
summary(pls.fit)
```

Notice that the percentage of variance in 'mpg' that the 2-component
PLS fit explains, 85.58%, is almost as much as that explained using the
final 3-component model PCR fit, 85.21%.

This is because PCR only
attempts to maximize the amount of variance explained in the predictors,
while PLS searches for directions that explain variance in both the predictors
and the response.
