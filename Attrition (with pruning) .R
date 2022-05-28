pacman::p_load(pacman, rio, tidyverse)
attrition = read.csv("Documents/Data_Mining/Group_Project/Attrition.csv")
#EDA
str(attrition)
summary(attrition)
#convert categorical variables to factor 
#split data
index <- sample(nrow(attrition),nrow(attrition)*0.90)
attrition_train = attrition[index,]
attrition_test = attrition[-index,]
#model fitting
attrition_glm0<- glm(Attrition ~., family=binomial, data=attrition_train)
summary(attrition_glm0)
mean(attrition_train$Attrition) #the mean
var(attrition_train$Attrition) #the variance
hist(attrition_train$Attrition) #the histogram
hist(attrition_train$PercentSalaryHike) #histogram
hist(attrition_train$YearsAtCompany) #histogram
boxplot(Age ~ Attrition, family = binomial, data = attrition_train, main = "Impact of other vaiables on Attrition", xlab = "Attrition", ylab = "Age") #the boxplot
boxplot(PercentSalaryHike ~ Attrition, family = binomial, data = attrition_train, main = "Impact of other vaiables on Attrition", xlab = "Attrition", ylab = "PercentSalaryHike") #the boxplot

#classification tree technique 
#with pruning the tree
library(rpart) 
library(rpart.plot)
attrition_rpart <- rpart(formula = Attrition ~ . , data = attrition_train, method =
                           "class", parms = list(loss=matrix(c(0,20,1,0), nrow = 2)), cp = 0.0020) #after running the code with original cp value of 0.001, select the cp value that falls under horizontal line 
attrition_rpart
prp(attrition_rpart, digits = 4, extra = 1)
plotcp(attrition_rpart)
printcp(attrition_rpart)

#in-sample prediction
pcut = 0.5 # default
cost1<-function(r, pi) mean( ((r==0)&(pi>pcut)) | ((r==1)&(pi<pcut)) )
cost1(attrition_train$Attrition, predict(attrition_rpart, attrition_train, type="prob")) #define symmetric cost function

cost <- function(r, phat){
  weight1 <- 20
  weight0 <- 1
  pcut <- weight0/(weight1+weight0)
  c1 <- (r==1)&(phat<pcut) #logical vector - true if actual 1 but predict 0 
  c0 <-(r==0)&(phat>pcut) #logical vector - true if actual 0 but predict 1 
  return(mean(weight1*c1+weight0*c0))}
cost(attrition_train$Attrition, predict(attrition_rpart, attrition_train, type="prob")) #define asymmetric cost function

attrition_train.pred.tree <- predict(attrition_rpart, attrition_train, type="class") #based on the default pcut value which is 0.5
table(attrition_train$Attrition, attrition_train.pred.tree, dnn=c("Truth","Predicted"))

#(ii)Test the out-of-sample performance. Using tree model built from (i) on the training data, test with the remaining 20% testing data. Report the out-of-sample AUC, misclassification cost and misclassification cost with symmetric cost and asymmetric cost function. 
cost1(attrition_test$Attrition, predict(attrition_rpart, attrition_test, type="prob")) #symmetric cost
cost(attrition_test$Attrition, predict(attrition_rpart, attrition_test, type="prob")) #asymmetric cost 

#Probability of getting 1
attrition_test_prob_rpart = predict(attrition_rpart, attrition_test, type="prob")
library(ROCR)
#ROC curve
pred = prediction(attrition_test_prob_rpart[,2], attrition_test$Attrition)
perf = performance(pred, "tpr", "fpr")
plot(perf, colorize=TRUE)
#AUC
slot(performance(pred, "auc"), "y.values")[[1]]

attrition_test_pred_rpart = as.numeric(attrition_test_prob_rpart[,2] > 0.5)
table(attrition_test$Attrition, attrition_test_pred_rpart, dnn=c("Truth","Predicted"))

#new data mining tools 
attrition_glm_back <- step(attrition_glm0) #backward selection

nullmodel = glm(Attrition ~ 1, data = attrition_train)
fullmodel = glm(Attrition ~ ., data = attrition_train)
model.step <- step(nullmodel, scope = list(lower = nullmodel,
                                           upper = fullmodel), direction = "forward") #forward selection

#boostrap technique
library(boot)
boot.fn=function(attrition_train,index)
return(coef(glm(Attrition~ OverTime + JobRole + MaritalStatus + EnvironmentSatisfaction + 
                  JobSatisfaction + JobInvolvement + BusinessTravel + YearsWithCurrManager + 
                  EducationField + DistanceFromHome + Age + NumCompaniesWorked + 
                  YearsSinceLastPromotion + WorkLifeBalance + RelationshipSatisfaction + 
                  TrainingTimesLastYear + Gender + TotalWorkingYears + YearsInCurrentRole + 
                  YearsAtCompany + DailyRate + StockOptionLevel, family = binomial, data = attrition_train, subset = index))) #the best model

boot.fn(attrition, 1:2646)
set.seed(1)
boot.fn(attrition_train,sample(2646,2646,replace=T))
boot.fn(attrition_train,sample(2646,2646,replace=T))
boot(data = attrition_train, statistic = boot.fn, R = 1000)
summary(glm(Attrition~OverTime + JobRole + MaritalStatus + EnvironmentSatisfaction + 
              JobSatisfaction + JobInvolvement + BusinessTravel + YearsWithCurrManager + 
              EducationField + DistanceFromHome + Age + NumCompaniesWorked + 
              YearsSinceLastPromotion + WorkLifeBalance + RelationshipSatisfaction + 
              TrainingTimesLastYear + Gender + TotalWorkingYears + YearsInCurrentRole + 
              YearsAtCompany + DailyRate + StockOptionLevel, family = binomial, attrition_train))$coef

#lasso regression
x= model.matrix(Attrition ~., family = binomial, attrition_train)[,-1]
y= attrition_train$Attrition
library(glmnet)
grid= 10^seq(10,-2, length=100)
dim(coef(lasso.mod))
set.seed (1)
train=sample(1:nrow(x), nrow(x)/2)
test=(-train)
y.test=y[test]
lasso.mod=glmnet(x[train ,],y[train],alpha=1,lambda=grid)
dim(coef(lasso.mod))
plot(lasso.mod)
set.seed (1)
cv.out=cv.glmnet(x[train ,],y[train], alpha=1)
plot(cv.out)
bestlam=cv.out$lambda.min
lasso.pred=predict(lasso.mod,s=bestlam ,newx=x[test,])
mean((lasso.pred-y.test)^2)
out=glmnet(x,y,alpha=1,lambda=grid)
lasso.coef=predict(out,type="coefficients",s=bestlam)[1:34,]
lasso.coef
lasso.coef[lasso.coef!=0]

#comparison between lasso and forward selection 
#forward selection in-sample and out-of-sample prediction 
attrition_glm1 = glm(Attrition~OverTime + JobRole + MaritalStatus + EnvironmentSatisfaction + 
                       JobSatisfaction + JobInvolvement + BusinessTravel + YearsWithCurrManager + 
                       EducationField + DistanceFromHome + Age + NumCompaniesWorked + 
                       YearsSinceLastPromotion + WorkLifeBalance + RelationshipSatisfaction + 
                       TrainingTimesLastYear + Gender + TotalWorkingYears + YearsInCurrentRole + 
                       YearsAtCompany + DailyRate + StockOptionLevel, family=binomial, data=attrition_train) #the best model
#in-sample prediction with the best model
pred_glm1_train <- predict(attrition_glm1, type="response")
#ROC curve
library(ROCR)
pred <- prediction(pred_glm1_train, attrition_train$Attrition) 
perf <- performance(pred, "tpr", "fpr")
plot(perf, colorize=TRUE)
#AUC
unlist(slot(performance(pred, "auc"), "y.values"))
#specify objective function
pcut <- 0.5
cost2 <- function(r, pi, pcut){
  weight1 <- 20
  weight0 <- 1
  c1 <- (r==1)&(pi<pcut) #logical vector - true if actual 1 but predict 0 
  c0 <-(r==0)&(pi>pcut) #logical vector - true if actual 0 but predict 1 
  return(mean(weight1*c1+weight0*c0))} #asymmetric cost 
cost2

#cutoff value
pred_resp <- predict(attrition_glm1,type="response")
table(attrition_train$Attrition, (pred_glm1_train > 0.5)*1, dnn=c("Truth","Predicted"))
#(iii) Test the out-of-sample performance. Using final logistic linear model built from (ii) on the 80% of original data, test with the remaining 20% testing data.  (Try predict() function in R.) Report out-of-sample AUC and misclassification rate. 
#out-of-sample prediction with the best model
pred_glm1_test<- predict(attrition_glm1, newdata = attrition_test, type="response")
#ROC curve
pred <- prediction(pred_glm1_test, attrition_test$Attrition)
perf <- performance(pred, "tpr", "fpr")
plot(perf, colorize=TRUE)
#AUC
unlist(slot(performance(pred, "auc"), "y.values"))
#specify objective function
pcut <- 0.5
cost2 <- function(r, pi, pcut){
  weight1 <- 20
  weight0 <- 1
  c1 <- (r==1)&(pi<pcut) #logical vector - true if actual 1 but predict 0 
  c0 <-(r==0)&(pi>pcut) #logical vector - true if actual 0 but predict 1 
  return(mean(weight1*c1+weight0*c0))} #asymmetric cost 

#cutoff value 
pred_resp <- predict(attrition_glm1, data=attrition_test, type="response")
table(attrition_test$Attrition, (pred_glm1_test > 0.5)*1, dnn=c("Truth","Predicted"))

#lasso in-sample and out-of-sample prediction 
attrition_glm2 = glm(Attrition~ Age + BusinessTravel+ DailyRate + Department+ DistanceFromHome + Education + EnvironmentSatisfaction + Gender + JobInvolvement + JobLevel + JobRole+
                    JobSatisfaction + MaritalStatus+ NumCompaniesWorked, family=binomial, data=attrition_train) #the best model
#in-sample prediction with the best model
pred_glm2_train <- predict(attrition_glm2, type="response")
#ROC curve
library(ROCR)
pred <- prediction(pred_glm2_train, attrition_train$Attrition) 
perf <- performance(pred, "tpr", "fpr")
plot(perf, colorize=TRUE)
#AUC
unlist(slot(performance(pred, "auc"), "y.values"))
#specify objective function
pcut <- 1/(20+1)
cost2 <- function(r, pi, pcut){
  weight1 <- 20
  weight0 <- 1
  c1 <- (r==1)&(pi<pcut) #logical vector - true if actual 1 but predict 0 
  c0 <-(r==0)&(pi>pcut) #logical vector - true if actual 0 but predict 1 
  return(mean(weight1*c1+weight0*c0))} #asymmetric cost 
cost2

#cutoff value
pred_resp <- predict(attrition_glm2,type="response")
table(attrition_train$Attrition, (pred_glm2_train > 0.5)*1, dnn=c("Truth","Predicted"))
#(iii) Test the out-of-sample performance. Using final logistic linear model built from (ii) on the 80% of original data, test with the remaining 20% testing data.  (Try predict() function in R.) Report out-of-sample AUC and misclassification rate. 
#out-of-sample prediction with the best model
pred_glm2_test <- predict(attrition_glm2, newdata = attrition_test, type="response")
#ROC curve
pred <- prediction(pred_glm1_test, attrition_test$Attrition)
perf <- performance(pred, "tpr", "fpr")
plot(perf, colorize=TRUE)
#AUC
unlist(slot(performance(pred, "auc"), "y.values"))
#specify objective function
pcut <- 1/(20+1)
cost2 <- function(r, pi, pcut){
  weight1 <- 20
  weight0 <- 1
  c1 <- (r==1)&(pi<pcut) #logical vector - true if actual 1 but predict 0 
  c0 <-(r==0)&(pi>pcut) #logical vector - true if actual 0 but predict 1 
  return(mean(weight1*c1+weight0*c0))} #asymmetric cost 

#cutoff value 
pred_resp <- predict(attrition_glm2, data=attrition_test, type="response")
table(attrition_test$Attrition, (pred_glm2_test > 0.5)*1, dnn=c("Truth","Predicted"))
