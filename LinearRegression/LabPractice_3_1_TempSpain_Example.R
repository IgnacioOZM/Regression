#################################################################################
##############       LabPractice 3.1 Regression     ############################
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
fdataINI <- read.csv("TemperaturesSpainSmall.csv", sep = ";")
fdataTOT <- fdataINI[1:1000,] #Select 1000 for class purposes
fdataTOT = na.omit(fdataTOT) #Eliminate NA

# Contents of the dataset
str(fdataTOT)

# Output variable (TMIN MADRID-RETIRO)
summary(fdataTOT$TMIN230)
hist(fdataTOT$TMIN230, nclass = 40)

# Subsets variables
library(dplyr) # select, grouping levels in factor
fdataTmax = select(fdataTOT, "WEEKDAY","MONTH", starts_with("TMAX"))
str(fdataTmax)
fdata = select(fdataTOT, "WEEKDAY","MONTH","TMIN230","TMAX230","TMAX229","TMAX237",
                "TMAX417","TMAX2969","TMAX3910","TMAX3918","TMAX3959")
str(fdata)

#correlation plot of numeric variables
numvars <- sapply(fdata, class) %in% c("integer","numeric")
C <- cor(fdata[,numvars])
corrplot(C, method = "circle")

## Exploratory analysis -------------------------------------------------------------------------------------
ggpairs(fdata,aes( alpha = 0.3))
PlotDataframe(fdata,output.name = "TMIN230")


## Filter outliers
fdata$TMAX417[fdata$TMAX417>35] = NA
fdata = na.omit(fdata) #Eliminate NA, i.e. the outliers



## Model training ------------------------------------------------------------------------------

## Divide the data into training and validation sets ---------------------------------------------------
set.seed(150) #For replication
#create random 80/20 % split
trainIndex <- createDataPartition(fdata$TMIN230,      #output variable. createDataPartition creates proportional partitions
                                  p = 0.8,      #split probability for training
                                  list = FALSE, #Avoid output as a list
                                  times = 1)    #only one partition
#obtain training and validation sets
fTR <- fdata[trainIndex,]
fTR_eval <- fTR #
fTV <- fdata[-trainIndex,]


## Initialize trainControl (FOR ALL MODELS) -----------------------------------------------------------------------
#Use resampling for measuring generalization error
#K-fold with 10 folds
ctrl_tune <- trainControl(method = "cv",                     
                          number = 10,
                          summaryFunction = defaultSummary,    #Performance summary for comparing models in hold-out samples.
                          returnResamp = "final",              #Return final information about resampling
                          savePredictions = TRUE)              #save predictions

#==================================================================
##FIRST model for 230 MADRID-RETIRO using WEEKDAY, MONTH and TMAX230 (highly correlated) 
#==================================================================
set.seed(150) #For replication
lm1.fit = train(form = TMIN230 ~ WEEKDAY + MONTH + TMAX230, 
               data = fTR, 
               method = "lm", #Linear model
               #tuneGrid = data.frame(intercept = TRUE), 
               preProcess = c("center","scale"),
               trControl = ctrl_tune, 
               metric = "RMSE")
lm1.fit #information about the resampling settings
summary(lm1.fit)  #information about the model trained

#Evaluate the model with training sets and diagnosis
fTR_eval$lm_pred1 = predict(lm1.fit,  newdata = fTR)  

PlotModelDiagnosis(fTR[,c("WEEKDAY", "MONTH", "TMAX230")]
                      , fTR$TMIN230, fTR_eval$lm_pred1,
                   together = TRUE)

# Residuals vs WEEKDAY suggests ...
# Residuals vs MONTH suggests ...
#    MONTH is used in the model (int) as a numeric variable ??
fTR$MONTHc = as.factor(fTR$MONTH);


# Residuals by month
PlotModelDiagnosis(fTR[,c("WEEKDAY", "MONTHc", "TMAX230")]
                   , fTR$TMIN230, fTR_eval$lm_pred1,
                   together = TRUE)


#==================================================================
## SECOND model for 230 MADRID-RETIRO using factor(MONTH) and TMAX230
#==================================================================

#Creating dummy variables by hand (other option)
# dummyModel <- dummyVars(MONTHc~., data = fdata, fullRank = TRUE)
# fdataDV <- as.data.frame(predict(dummyModel, fdata))
# fdataDV$TMIN230 <- fdata$TMIN230

set.seed(150) #For replication
lm2.fit = train(form = TMIN230 ~ MONTHc + TMAX230 , # as factor !
               data = fTR, 
               method = "lm", #Linear model
               #tuneGrid = data.frame(intercept = TRUE), 
               #preProcess = c("center","scale"),
               trControl = ctrl_tune, 
               metric = "RMSE")
lm2.fit #information about the resampling settings
summary(lm2.fit)  #information about the model trained

#Evaluate the model with training sets and diagnosis
fTR_eval$lm_pred2 = predict(lm2.fit,  newdata = fTR)  

PlotModelDiagnosis(fTR[,c("MONTHc", "TMAX230")]
                   , fTR$TMIN230, fTR_eval$lm_pred2,
                   together = TRUE)

ggplot(fTR_eval)+geom_point(aes(x=TMAX230, y=TMIN230), alpha = 0.3)

# The model has 12 regression lines with the same slope (one by month)
fTR_eval$MONTHc = fTR$MONTHc
ggplot(fTR_eval)+geom_point(aes(x=TMAX230, y=lm_pred2, color=MONTHc), alpha = 0.3)

#plot the real point and estimations with months
ggplot(fTR_eval)+geom_point(aes(x=TMAX230, y=TMIN230), alpha = 0.2)+
  geom_point(aes(x=TMAX230, y=lm_pred2, color=MONTHc), alpha = 0.5) 


# pvalues for some levels of MONTHc suggest ...

#==================================================================
## THIRD model for 230 MADRID-RETIRO using INTERACTION between GroupsMONTHc and TMAX230
#==================================================================

# Removing some levels of MONTHc means grouping them in sets
levels(fTR$MONTHc) # 12 different values

# Create new factor grouping months 
fTR$MONTHg = fTR$MONTHc
#                        (1,2,3,4,5,6,7,8,9,10,11,12)
levels(fTR$MONTHg) <- c(1,1,1,1,5,6,7,8,9,10,11, 1) # INV/VER


set.seed(150) #For replication
lm3.fit = train(form = TMIN230 ~ MONTHg + TMAX230,  # GROUPING levels of MONTHc
                data = fTR, 
                method = "lm", #Linear model
                #tuneGrid = data.frame(intercept = TRUE), 
                preProcess = c("center","scale"),
                trControl = ctrl_tune, 
                metric = "RMSE")
lm3.fit #information about the resampling settings
summary(lm3.fit)  #information about the model trained

#Evaluate the model with training sets and diagnosis
fTR_eval$MONTHg = fTR$MONTHg
fTR_eval$lm_pred3 = predict(lm3.fit,  newdata = fTR)  

PlotModelDiagnosis(fTR[,c("TMAX230", "MONTHg")]
                      , fTR$TMIN230, fTR_eval$lm_pred3,
                   together = TRUE)


# The model has 8 regression lines with the same slope
ggplot(fTR_eval)+geom_point(aes(x=TMAX230, y=lm_pred3, color=MONTHg), alpha = 0.3)


#==================================================================
## FOURTH model for 230 MADRID-RETIRO using INTERACTION between MONTHg and TMAX230
#==================================================================

set.seed(150) #For replication
lm4.fit = train(form = TMIN230 ~ MONTHg*TMAX230, # as factor !
                data = fTR, 
                method = "lm", #Linear model
                #tuneGrid = data.frame(intercept = TRUE), 
                preProcess = c("center","scale"),
                trControl = ctrl_tune, 
                metric = "RMSE")
lm4.fit #information about the resampling settings
summary(lm4.fit)  #information about the model trained

#Evaluate the model with training sets and diagnosis
fTR_eval$lm_pred4 = predict(lm4.fit,  newdata = fTR)  

PlotModelDiagnosis(fTR[,c("MONTHg", "TMAX230")]
                      , fTR$TMIN230, fTR_eval$lm_pred4,
                   together = TRUE)

#The model has 8 regression lines with different slopes
ggplot(fTR_eval)+geom_point(aes(x=TMAX230, y=lm_pred4, color=MONTHg), alpha = 0.3)


# Possible quadratic effect for TMAX230?...

#==================================================================
## FIFTH model for 230 MADRID-RETIRO between MONTHg and TMAX230 and TMAX230^2
#==================================================================
set.seed(150) #For replication
lm5.fit = train(form = TMIN230 ~ MONTHg+poly(TMAX230,2,raw=TRUE), # force raw poly vals
                data = fTR, 
                method = "lm", #Linear model
                #tuneGrid = data.frame(intercept = TRUE), 
                preProcess = c("center","scale"),
                trControl = ctrl_tune, 
                metric = "RMSE")
lm5.fit #information about the resampling settings
summary(lm5.fit)  #information about the model trained

#Evaluate the model with training sets and diagnosis
fTR_eval$lm_pred5 = predict(lm5.fit,  newdata = fTR)  


PlotModelDiagnosis(fTR[,c("MONTHg", "TMAX230")]
                   , fTR$TMIN230, fTR_eval$lm_pred5,
                   together = TRUE)

#The model has 8 regression parabolas 
ggplot(fTR_eval)+geom_point(aes(x=TMAX230, y=lm_pred5, color=MONTHg), alpha = 0.3)



#-------------------------------------------------------------------------------------------------
#--------------------------- Cross-validation results ------------------------------------------
#-------------------------------------------------------------------------------------------------
transformResults <- resamples(list(
  lm1.MONTH=lm1.fit,
  lm2.MONTHc=lm2.fit,
  lm3.MONTHg=lm3.fit,
  lm4.Interaction=lm4.fit,
  lm5.poly=lm5.fit))
summary(transformResults)
dotplot(transformResults)


#-------------------------------------------------------------------------------------------------
#--------------------------- VALIDATION  results ------------------------------------------
#-------------------------------------------------------------------------------------------------

# add new input vars to validation
# Create new factor grouping months
fTV$MONTHc = as.factor(fTV$MONTH);
fTV$MONTHg = fTV$MONTHc
levels(fTV$MONTHg) <- c(1,1,1,1,5,6,7,8,9,10,11, 1) # INV/VER

#validation
fTV_eval = fTV
#Evaluate the models
fTV_eval$lm_pred1 = predict(lm1.fit,  newdata = fTV)  
fTV_eval$lm_pred2 = predict(lm2.fit,  newdata = fTV)  
fTV_eval$lm_pred3 = predict(lm3.fit,  newdata = fTV)  
fTV_eval$lm_pred4 = predict(lm4.fit,  newdata = fTV)  
fTV_eval$lm_pred5 = predict(lm5.fit,  newdata = fTV)  


## compare results -------------------------------------------------------------------------------
#R2
caret::R2(fTR_eval$lm_pred5,fTR_eval$TMIN230)
caret::R2(fTV_eval$lm_pred5,fTV_eval$TMIN230)

#MSE
caret::RMSE(fTR_eval$lm_pred5,fTR_eval$TMIN230)
caret::RMSE(fTV_eval$lm_pred5,fTV_eval$TMIN230)

