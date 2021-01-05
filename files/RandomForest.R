set.seed(9386311)
source("Helper.R")
library(randomForest) # For random forests
# Read in the Training Set
data = read.csv("train.csv")

### The function that fits random forests requires that our response
### variable be a factor. We need to make a copy of our dataset and
### use the factor() function on quality.
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

# CV (using the RF OOB error) within CV 
K = 5 # Number of folds
folds = read.csv("folds.csv" )

### Create container for CV errors
all.models = c("LR", "NN", "NB", "DT", "RF", "Boosting")
n.models = length(all.models)
CV.CEs = read.csv("CV-CEs.csv")
rownames(CV.CEs) = all.models
###########
### RF  ###
###########
# Tuning parameters for RF
all.mtrys = c(1)
all.nodesizes = c(1, 5, 15, 20, 25)
all.pars.rf = expand.grid(mtry = all.mtrys, nodesize = all.nodesizes)
n.pars = nrow(all.pars.rf)

M = 5 # Number of times to repeat RF fitting. I.e. Number of OOB errors


for (cv in 1:K){
  print(paste0(cv, " of ", K, " of the outermost CV loop"))
  
  
  ### Split data
  data.train = data[folds != cv,]
  data.valid = data[folds == cv,]
  Y.valid = data.valid$y
  
  ### Container to store OOB errors. This will be easier to read if we name
  ### the columns.
  all.OOB.rf = array(0, dim = c(M, n.pars))
  names.pars = apply(all.pars.rf, 1, paste0, collapse = "-")
  colnames(all.OOB.rf) = names.pars
  
  # data for tuning using OOB error
  data.train.rf = data.train
  Y.train.rf = data.train.rf$y
  
  for(i in 1:n.pars){
    ### Progress update
    print(paste0(i, " of ", n.pars))
    
    ### Get tuning parameters for this iteration
    this.mtry = all.pars.rf[i, "mtry"]
    this.nodesize = all.pars.rf[i, "nodesize"]
    
    for(j in 1:M){
      ### Fit RF, then get and store OOB errors
      this.fit.rf = randomForest(y ~ ., data = data.train.rf,
                                 mtry = this.mtry, nodesize = this.nodesize)
      
      pred.this.rf = predict(this.fit.rf)
      this.err.rf = mean(Y.train.rf != pred.this.rf)
      
      all.OOB.rf[j, i] = this.err.rf
    }
  }
  
  ### Make a regular and relative boxplot
  boxplot(all.OOB.rf, las=2, main = "OOB Boxplot")
  rel.OOB.rf = apply(all.OOB.rf, 1, function(W) W/min(W))
  boxplot(t(rel.OOB.rf), las=2,  # las sets the axis label orientation
          main = "Relative OOB Boxplot")
  
  
  # Find hyperparameters corresponding to the 
  # minimum average OOB error across the M=5 repetitions 
  ave.err = apply(all.OOB.rf, 2, mean)
  best.ind = which.min(ave.err)
  best.mtry = all.pars.rf[best.ind,"mtry"]
  best.nodesize = all.pars.rf[best.ind,"nodesize"]
  
  ### Let's fit this RF model to the entire training fold
  fit.rf.best = randomForest(y ~ ., data = data.train,
                        mtry = best.mtry, nodesize = best.nodesize)
  
  ### Evaluate performance on the validation set
  pred.rf = predict(fit.rf.best, data.valid)
  table.rf =  table(Y.valid, pred.rf, dnn = c("Obs", "Pred"))
  CV.CEs["RF", cv] = mean(Y.valid != pred.rf)
  print(CV.CEs)
  
}

#save the CEs
write.csv(CV.CEs, file='CV-CEs.csv', row.names=F)
CV.CEs = read.csv("CV-CEs.csv")
rownames(CV.CEs) = all.models











