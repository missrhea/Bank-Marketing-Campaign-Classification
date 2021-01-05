set.seed(9386311)
library(klaR)   # For naive Bayes
source("Helper.R")

# Read in the Training Set
data = read.csv("train.csv")

### The function that fits Naive Bayes Model requires that our response
### variable be a factor
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

# Create a numerical version of variables for methods that need numbers
data$job <- as.numeric((data$job))
data$marital <- as.numeric((data$marital))
data$education <- as.numeric((data$education))
data$default <- as.numeric((data$default))
data$housing <- as.numeric((data$housing))
data$loan <- as.numeric((data$loan))
data$poutcome <- as.numeric((data$poutcome))

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
### NB  ###
###########

for (cv in 1:K){
  
  print(paste0(cv, " of ", K, " of the outermost CV loop"))
  
  ### Split data
  data.train = data[folds != cv,]
  data.valid = data[folds == cv,]
  X.train = data.train[,-16]
  X.valid = data.valid[,-16]
  Y.train = data.train$y
  Y.valid = data.valid$y
  # fit the model with usekernel=T so that it estimates the densities (instead
  # of assuming a normal distribution for the explanatories)
  fit.NB = NaiveBayes(X.train, Y.train, usekernel = T)
  
  # ## We can plot the kernel density estimates.
  # par(mfrow = c(2,3)) # Set plotting to 2x3
  # plot(fit.NB)
  # 1 # Dummy line because the plot function for NaiveBayes models takes the
  # next line of code as an extra input. It's weird. Just keep it here.
  # par(mfrow = c(1,1)) # Reset plotting to 1x1
  
  pred.NB.raw = predict(fit.NB, X.valid)
  pred.NB = pred.NB.raw$class
  table.NB = table(Y.valid, pred.NB, dnn = c("Obs", "Pred"))
  CV.CEs["NB", cv] = mean(Y.valid != pred.NB)
  print(CV.CEs)
}

#save the CEs
write.csv(CV.CEs, file='CV-CEs.csv', row.names=F)