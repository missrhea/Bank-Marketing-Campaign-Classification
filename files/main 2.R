set.seed(9386311)
source("Helper.R")

# Read in the Training Set
data = read.csv("train.csv")
head(data)
summary(data)
# Variable "y" is the response variable. Convert the character to factor.
class(data$y)
levels(data$y)
data$y = as.factor(data$y)
# check
class(data$y)
levels(data$y)
# Convert the character to factor for explanatory variables
data$job = as.factor(data$job)
data$marital = as.factor(data$marital)
data$education = as.factor(data$education)
data$default = as.factor(data$default)
data$housing = as.factor(data$housing)
data$loan = as.factor(data$loan)
data$poutcome = as.factor(data$poutcome)
# check
head(data)
summary(data)

d = data
# CV within CV OR nested CV

K = 5 # Number of folds
folds = read.csv("folds.csv" )

### Create container for CV errors
all.models = c("LR", "NN", "NB", "DT", "RF", "Boosting")
n.models = length(all.models)
CV.CEs = array(NA, dim = c(n.models, K))
rownames(CV.CEs) = all.models

###########
### LR  ###
###########

for(cv in 1:K){
  ### Print a status update
  print(paste0(cv, " of ", K, " of the outermost CV loop"))
  
  ### Split data
  data.train = data[folds != cv,]
  data.valid = data[folds == cv,]
  Y.valid = data.valid$y
  n.train = nrow(data.train)

  library(glmnet) # For logistic regression and LASSO
  # Process the data for Logistic Regression & LASSO with glmnet package

  # Create a numerical version of variables for methods that need numbers
  data.train$job <- as.numeric((data.train$job))
  data.train$marital <- as.numeric((data.train$marital))
  data.train$education <- as.numeric((data.train$education))
  data.train$default <- as.numeric((data.train$default))
  data.train$housing <- as.numeric((data.train$housing))
  data.train$loan <- as.numeric((data.train$loan))
  data.train$poutcome <- as.numeric((data.train$poutcome))

  data.valid$job <- as.numeric((data.valid$job))
  data.valid$marital <- as.numeric((data.valid$marital))
  data.valid$education <- as.numeric((data.valid$education))
  data.valid$default <- as.numeric((data.valid$default))
  data.valid$housing <- as.numeric((data.valid$housing))
  data.valid$loan <- as.numeric((data.valid$loan))
  data.valid$poutcome <- as.numeric((data.valid$poutcome))
  
  # Standardized the five social & economic variables
  data.train.raw = data.train
  data.train[,c(11,12,13,14,15)] = scale.1(data.train[,c(11,12,13,14,15)], data.train.raw[,c(11,12,13,14,15)])
  # use the training set mean & sd to standardize the validation set
  data.valid[,c(11,12,13,14,15)] = scale.1(data.valid[,c(11,12,13,14,15)], data.train.raw[,c(11,12,13,14,15)])

  ### Create copies of our datasets and rescale for logistic regression
  data.train.scale = data.train
  data.valid.scale = data.valid
  data.train.scale[,-16] = rescale(data.train.scale[,-16], data.train[,-16])
  data.valid.scale[,-16] = rescale(data.valid.scale[,-16], data.train[,-16])

  X.train.scale = as.matrix(data.train.scale[,-16])
  Y.train = data.train.scale[,16]
  X.valid.scale = as.matrix(data.valid.scale[,-16])
  Y.valid = data.valid.scale[,16]

  fit.log.glmnet = glmnet(X.train.scale, Y.train, family = "multinomial")
  pred.log.glmnet = predict(fit.log.glmnet, X.valid.scale, type = "class", s = 0)
  table.LR = table(Y.valid, pred.log.glmnet, dnn = c("Observed", "Predicted"))
  CV.CEs["LR", cv] = mean(Y.valid != pred.log.glmnet)
  print(CV.CEs)
  
}


#save the CEs
write.csv(CV.CEs, file='CV-CEs.csv', row.names=F)
CV.CEs = read.csv("CV-CEs.csv")
rownames(CV.CEs) = all.models

###########
### NN  ###
###########
### Tuning parameter values for NN
all.sizes = c(1, 3, 6, 10)
all.decays = c(0, 0.001, 0.01, 0.1, 1)
all.pars.nn = expand.grid(size = all.sizes, decay = all.decays)
n.pars.nn = nrow(all.pars.nn)
par.names.nn = apply(all.pars.nn, 1, paste, collapse = "-")

H.nn = 2   # Number of times to repeat CV
K.nn = 5   # Number of folds for CV
M.nn = 20  # Number of times to re-run nnet() at each iteration

for(cv in 1:K){
  ### Print a status update
  print(paste0(cv, " of ", K, " of the outermost CV loop"))
  
  ### Split data
  data.train = data[folds != cv,]
  data.valid = data[folds == cv,]
  Y.valid = data.valid$y
  n.train = nrow(data.train)
  
  # Create a numerical version of variables for methods that need numbers
  data.train$job <- as.numeric((data.train$job))
  data.train$marital <- as.numeric((data.train$marital))
  data.train$education <- as.numeric((data.train$education))
  data.train$default <- as.numeric((data.train$default))
  data.train$housing <- as.numeric((data.train$housing))
  data.train$loan <- as.numeric((data.train$loan))
  data.train$poutcome <- as.numeric((data.train$poutcome))
  
  data.valid$job <- as.numeric((data.valid$job))
  data.valid$marital <- as.numeric((data.valid$marital))
  data.valid$education <- as.numeric((data.valid$education))
  data.valid$default <- as.numeric((data.valid$default))
  data.valid$housing <- as.numeric((data.valid$housing))
  data.valid$loan <- as.numeric((data.valid$loan))
  data.valid$poutcome <- as.numeric((data.valid$poutcome))
  
  
  library(nnet)        # For neural networks
  ### Container for CV misclassification rates. Need to include room for
  ### H*K CV errors
  CV.misclass.nn = array(0, dim = c(H.nn*K.nn, n.pars.nn))
  colnames(CV.misclass.nn) = par.names.nn
  
  ### Tuning of the NN starts here
  for(h in 1:H.nn) {
    ### Get all CV folds
    folds.nn = get.folds(n.train, K.nn)
    
    for (i in 1:K.nn) {
      print(paste0(h, "-", i, " of ", H.nn, "-", K.nn))
      
      ### Split training set according to fold i
      data.train.inner = data.train[folds.nn != i, ]
      data.valid.inner = data.train[folds.nn == i, ]
      
      # Standardized the five social & economic variables
      data.train.raw = data.train.inner
      data.train.inner[,c(11,12,13,14,15)] = scale.1(data.train.inner[,c(11,12,13,14,15)], data.train.raw[,c(11,12,13,14,15)])
      # use the training set mean & sd to standardize the validation set
      data.valid.inner[,c(11,12,13,14,15)] = scale.1(data.valid.inner[,c(11,12,13,14,15)], data.train.raw[,c(11,12,13,14,15)])
      
      ### Separate response from predictors
      Y.train.inner = data.train.inner[, 16]
      X.train.inner.raw = data.train.inner[, -16]
      Y.valid.inner = data.valid.inner[, 16]
      X.valid.inner.raw = data.valid.inner[, -16]
      
      ### Transform predictors and response for nnet()
      X.train.inner = rescale(X.train.inner.raw, X.train.inner.raw)
      X.valid.inner = rescale(X.valid.inner.raw, X.train.inner.raw)
      Y.train.inner.num = class.ind(factor(Y.train.inner))
      Y.valid.inner.num = class.ind(factor(Y.valid.inner))
      
      for (j in 1:n.pars.nn) {
        print(paste0(j, " of ", n.pars.nn))
        ### Get parameter values
        this.size = all.pars.nn[j, "size"]
        this.decay = all.pars.nn[j, "decay"]
        
        ### Get ready to re-fit NNet with current parameter values
        CE.best = Inf
        
        ### Re-run nnet() M times and keep the one with best sMSE
        for (l in 1:M.nn) {
          this.nnet = nnet(
            X.train.inner,
            Y.train.inner.num,
            size = this.size,
            decay = this.decay,
            maxit = 10000,
            softmax = T,
            trace = F
          )
          this.MSE = this.nnet$value
          if (this.MSE < CE.best) {
            nnet.best = this.nnet
            CE.best = this.MSE
          }
        }
        
        ### Get CV misclassification rate for chosen nnet()
        pred.nnet.best = predict(nnet.best, X.valid.inner, type = "class")
        this.mis.CV = mean(Y.valid.inner != pred.nnet.best)
        
        ### Store this CV error. Be sure to put it in the correct row
        ind.row = (h - 1) * K.nn + i
        CV.misclass.nn[ind.row, j] = this.mis.CV
      }
    }
  }
  ave.err = apply(CV.misclass.nn, 2, mean)
  best.ind = which.min(ave.err)
  best.size = all.pars.nn[best.ind,"size"]
  best.decay = all.pars.nn[best.ind,"decay"]
  
  # Standardized the five social & economic variables
  data.train.raw = data.train
  data.train[,c(11,12,13,14,15)] = scale.1(data.train[,c(11,12,13,14,15)], data.train.raw[,c(11,12,13,14,15)])
  # use the training set mean & sd to standardize the validation set
  data.valid[,c(11,12,13,14,15)] = scale.1(data.valid[,c(11,12,13,14,15)], data.train.raw[,c(11,12,13,14,15)])
  
  ### Separate response from predictors
  Y.train = data.train[, 16]
  X.train.raw = data.train[, -16]
  Y.valid = data.valid[, 16]
  X.valid.raw = data.valid[, -16]
  
  ### Transform predictors and response for nnet()
  X.train = rescale(X.train.raw, X.train.raw)
  X.valid = rescale(X.valid.raw, X.train.raw)
  Y.train.num = class.ind(factor(Y.train))
  Y.valid.num = class.ind(factor(Y.valid))
  
  ### Let's fit this NNet model to the entire training set. 
  ### Remember to re-run nnet() (since we're just fitting one model, we can
  ### afford more re-runs)
  CE.best = Inf
  for(l in 1:(M.nn)){
    fit.nnet = nnet(X.train, Y.train.num, size = best.size, decay = best.decay, 
                    maxit = 1000, softmax = T, trace = F)
    this.MSE = fit.nnet$value
    if(this.MSE < CE.best){
      nnet.best = fit.nnet
      CE.best = this.MSE
    }
  }
  
  ### Evaluate performance on the validation set
  pred.nnet = predict(nnet.best, X.valid, type = "class")
  table.NN =  table(Y.valid, pred.nnet, dnn = c("Obs", "Pred"))
  CV.CEs["NN", cv] = mean(Y.valid != pred.nnet)
  print(CV.CEs)
  
}

# Boxplot of NN hyperparameters
arr = CV.misclass.nn
boxplot(arr, las = 2)
rel.CV.misclass2 = apply(arr, 1, function(W) W/min(W))
boxplot(t(rel.CV.misclass2), las=2, main="Boxplot of NN hyperparameters")
#boxplot(t(rel.CV.misclass2), las=2, ylim = c(1, 1.3))

#save the CEs
write.csv(CV.CEs, file='CV-CEs.csv', row.names=F)
CV.CEs = read.csv("CV-CEs.csv")
rownames(CV.CEs) = all.models



















# save( OOB.MSPEs, file='OOB.MSPEs.Rdata', row.names=F, col.names=F)
# write.csv(OOB.MSPEs, file='OOB.MSPEs.csv', row.names=F, col.names=F)
# save( MSPEs.nn, file='MSPEs.nn.Rdata', row.names=F, col.names=F)
# write.csv(MSPEs.nn, file='MSPEs.nn.csv', row.names=F, col.names=F)
# save( MSPEs.boosting, file='MSPEs.boosting.Rdata', row.names=F, col.names=F)
# write.csv(MSPEs.boosting, file='MSPEs.boosting.csv', row.names=F, col.names=F)
# save( MSPEs.ppr, file='MSPEs.ppr.Rdata', row.names=F, col.names=F)
# write.csv(MSPEs.ppr, file='MSPEs.ppr.csv', row.names=F, col.names=F)


### Make a boxplot of MSPEs for the different number of terms
boxplot(t(CV.CEs), main="MSPEs for 5 fold CV")

### Make a boxplot of RMSPEs for the different number of terms
CV.RMSPEs = apply(CV.MSPEs, 2, function(W) W/min(W))
boxplot(t(CV.RMSPEs), main="Relative Errors for 5 fold CV")
