#################################################################################
##############           Insurance dataset          ############################
##############     ----------- solution ---------   ############################
#################################################################################

## Set working directory -------------------------------------------------------------------------
x <-dirname(rstudioapi::getActiveDocumentContext()$path)
setwd(x)
rm(list = ls())
cat("\014")

## Load libraries --------------------------------------------------------------------------------
library(caret)
library(ggplot2)
library(GGally)
library(leaps)
library(glmnet)
library(pls)
library(car)
library(corrplot)
library(MLTools)


## LoadData 
fdata <- read.csv("insurance.csv",sep = ";")

## Exploratory analysis -------------------------------------------------------------------------------------
# - Types of variables
# - Identification of NA
# - Identification of outliers
# - Relations between input variables
# - Relation between the inputs and the output

# Contents of the dataset
str(fdata)

#Children has whole values from 0-5. 
#The the order has meaning (i.e. more children could mean greater charges) 
#It could be factor or numeric. If we choose factor, the model will be more detailed.
fdata$children <- as.factor(fdata$children)


summary(fdata)
#we can see there are no outliers

#Analyze output variable
hist(fdata$charges)
#skewed right distribution. The ideal would be to have a normal distibution.


#Explore relationships between variables
ggpairs(fdata,aes( alpha = 0.3))
#we can see that there does not seem to be any relation between numeric inputs (age and bmi)
#and the distribution of those per category of factor variables seem normal.
#Doesnt seem to be any outliers in the data.


#correlation plot of numeric variables
numvars <- sapply(fdata, class) %in% c("integer","numeric")
C <- cor(fdata[,numvars])
corrplot::corrplot(C, method = "circle")
#We can confirm that there is not a high LINEAR correlation between numeric inputs. 
#we can also see that age is more LINEARLY correlated to the output than bmi.


#To analyze in greater detail the relation between inputs and output, use PlotDataframe function
PlotDataframe(fdata, output.name = "charges")
#We can observe serveral effects from these plots:
# - Smoker seems a relevant variable looking at the boxplots. If smoker=yes, charges are higher.
# - We confirm the linear correlation between age and charges, although we can see that there seems 
#   to be 3 different behaviors. These should be explained by a factor variable. Also, the effects
#   seem to be a displacement on the level of charges. Therefore, interactions might not be needed in this case.
# - We see that bmi is related to charges, but there also appear to be different behaviors.
#   In this case, the effect of bmi on charges changes slope between behaviors. Interaction might be needed.


#plotting bmi and ages against charges and grouping by smoker, 
#we can see that these different behavior are greatly explained by this variable
#(we have included regression lines per category)
ggplot(fdata,aes(x=age,y=charges, color = smoker))+geom_point()+geom_smooth(method="lm")
ggplot(fdata,aes(x=bmi,y=charges, color = smoker))+geom_point()+geom_smooth(method="lm")


## Model training ------------------------------------------------------------------------------

## Divide the data into training and test sets ---------------------------------------------------
set.seed(150) #For replication
#create random 80/20 % split
trainIndex <- createDataPartition(fdata$charges,      #output variable. createDataPartition creates proportional partitions
                                  p = 0.8,      #split probability for training
                                  list = FALSE, #Avoid output as a list
                                  times = 1)    #only one partition
#obtain training and test sets
fTR <- fdata[trainIndex,]
fTS <- fdata[-trainIndex,]

#create dataset for storing evaluations
fTR_eval <- fTR
fTS_eval <- fTS


## Initialize trainControl -----------------------------------------------------------------------
#Use resampling for measuring generalization error
#K-fold with 10 folds
ctrl_tune <- trainControl(method = "cv",                     
                          number = 10,
                          summaryFunction = defaultSummary,    #Performance summary for comparing models in hold-out samples.
                          returnResamp = "final",              #Return final information about resampling
                          savePredictions = TRUE)              #save predictions



## Process for model identification in regression:
# 1) Train model
# 2) Analyze variable importance
# 3) Analyze multicolinearity
# 4) Analyze residuals against input variables (All plots should be centered in 0)
# 5) Identify nonlinearities and interactions
# Repeat from 1) with selected variables and nonlinear and interaction effects


## Linear Regression -------------------------------------------------------------------------------------------

#Lets begin with an initial model using all input variables.
#It is usually recomended to center and scale variables.
#For teaching purposes, this transformation will not be applied.
set.seed(150) #For replication
lm.fit <- train(form = charges~.,
               data = fTR, 
               method = "lm", #Linear model
               tuneGrid = data.frame(intercept = TRUE), 
               #preProcess = c("center","scale"),
               trControl = ctrl_tune, 
               metric = "RMSE")
lm.fit #information about the resampling settings

summary(lm.fit)  #information about the model trained


#Identify correlated variables
vif(lm.fit$finalModel)


#Evaluate the model with training sets and diagnosis
fTR_eval$lm_pred <- predict(lm.fit,  newdata = fTR)  
fTS_eval$lm_pred <- predict(lm.fit,  newdata = fTS)  

#Analysis of residuals
PlotModelDiagnosis(fTR, fTR$charges, fTR_eval$lm_pred,
                  together = TRUE)


ggplot(fTR_eval)+geom_point(aes(x=bmi,y=charges),alpha = 0.5)+
  geom_point(aes(x=bmi,y=lm_pred), alpha =0.5, color="red")


#Training and test errors
caret::R2(fTR_eval$lm_pred,fTR_eval$charges)
caret::R2(fTS_eval$lm_pred,fTS_eval$charges)
