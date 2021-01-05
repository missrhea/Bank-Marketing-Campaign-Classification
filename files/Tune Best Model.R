
# Tuning the NN model using 5 fold CV on the training set
set.seed(9386311)
source("Helper.R")
library(nnet)


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

# Create a numerical version of variables for methods that need numbers
data$job <- as.numeric((data$job))
data$marital <- as.numeric((data$marital))
data$education <- as.numeric((data$education))
data$default <- as.numeric((data$default))
data$housing <- as.numeric((data$housing))
data$loan <- as.numeric((data$loan))
data$poutcome <- as.numeric((data$poutcome))

d = data

#For simplicity, rename data as "train.x" and "train.y"
train.x = data[,-16]
train.y.class = data[,16] 
train.y = class.ind(train.y.class)

# can set a different seed now since we are going to compute new folds
# and nnet method has variability anyway
set.seed(74375641)
#  Let's do R=2 reps of V=10-fold CV.
V=10
R=2 
n = nrow(train.x)
# Create the folds and save in a matrix
folds = matrix(NA, nrow=n, ncol=R)
for(r in 1:R){
  folds[,r]=floor((sample.int(n)-1)*V/n2) + 1
}

# Grid for tuning parameters and number of restarts of nnet
siz <- c(3,6,10)
dec <- c(0.01, 0.1, 0.5, 0.75, 1)
nrounds=10

# Prepare matrix for storing results: 
#   row = 1 combination of tuning parameters
#   column = 1 split
#   Add grid values to first two columns

Mis.cv = matrix(NA, nrow=length(siz)*length(dec), ncol=V*R+2)
Mis.cv[,1:2] = as.matrix(expand.grid(siz,dec))

# Start loop over all reps and folds.  
for (r in 1:R){ 
  for(v in 1:V){
    
    print(paste0(r, "-", v, " of ", R, "-", V, " of the outermost CV loop"))
    
    y.1 <- as.matrix(train.y[folds[,r]!=v,])
    x.1.unscaled <- as.matrix(train.x[folds[,r]!=v,]) 
    x.1 <- rescale(x.1.unscaled, x.1.unscaled) 
    
    #Test
    y.2 <- as.matrix(train.y[folds[,r]==v],)
    x.2.unscaled <- as.matrix(train.x[folds[,r]==v,])
    x.2 = rescale(x.2.unscaled, x.1.unscaled)
    
    # Start counter to add each model's misclassification to row of matrix
    qq=1
    # Start Analysis Loop for all combos of size and decay on chosen data set
    for(d in dec){
      for(s in siz){
        
        print(paste0("decay-size = ", d, "-", s, "  |   qq = ", qq))
        
        ## Restart nnet nrounds times to get best fit for each set of parameters 
        Mi.final <- 1
        #  check <- MSE.final
        for(i in 1:nrounds){
          nn <- nnet(y=y.1, x=x.1, size=s, decay=d, maxit=2000, softmax=TRUE, trace=FALSE)
          Pi <- predict(nn, newdata=x.1, type="class")
          Mi <- mean(Pi != train.y.class[folds[,r]!=v])
          
          if(Mi < Mi.final){ 
            Mi.final <- Mi
            nn.final <- nn
          }
        }
        pred.nn = predict(nn.final, newdata=x.2, type="class")
        Mis.cv[qq,(r-1)*V+v+2] = mean(pred.nn != as.factor(train.y.class[folds[,r]==v]))
        qq = qq+1
      }
    }
    print(paste0(Mis.cv))
  }
}
Mis.cv

#save the Mis.cv
write.csv(Mis.cv, file='Mis-cv.csv', row.names=F)
Mis.cv = read.csv("Mis-cv.csv")

# 7:21AM to 10:00 AM for r=1

(Micv = apply(X=Mis.cv[,-c(1,2)], MARGIN=1, FUN=mean))
(Micv.sd = apply(X=Mis.cv[,-c(1,2)], MARGIN=1, FUN=sd))
Micv.CIl = Micv - qt(p=.975, df=R*V-1)*Micv.sd/sqrt(R*V)
Micv.CIu = Micv + qt(p=.975, df=R*V-1)*Micv.sd/sqrt(R*V)
(all.cv = cbind(round(cbind(Micv,Micv.CIl, Micv.CIu),5)))
siz.dec <- paste("NN",Mis.cv[,1],"-",Mis.cv[,2])
#rownames(Mis.cv) = siz.dec



# Plot results. 
boxplot(x=all.cv, use.cols=FALSE,  names=siz.dec,
        las=2, main="MisC Rate boxplot for various NNs")

# Plot RELATIVE results. 
(all.cv = cbind(Mis.cv[,1:2],round(cbind(Micv,Micv.CIl, Micv.CIu),2)))
lowt = apply(all.cv, 2, min)
# all.cv[order(Micv),]

x11(pointsize=10)
# margin defaults are 5,4,4,2, bottom, left, top right
#  Need more space on bottom, so increase to 7.
par(mar=c(7,4,4,2))
boxplot(x=t(Mis.cv[,-c(1,2)])/lowt, las=2 ,names=siz.dec,
        main="Relative MisC Rate boxplot for various NNs",
        ylim = c(0.8, 1.3))

relMi = t(Mis.cv[,-c(1,2)])/lowt
(RRMi = apply(X=relMi, MARGIN=2, FUN=mean))
(RRMi.sd = apply(X=relMi, MARGIN=2, FUN=sd))
RRMi.CIl = RRMi - qt(p=.975, df=R*V-1)*RRMi.sd/sqrt(R*V)
RRMi.CIu = RRMi + qt(p=.975, df=R*V-1)*RRMi.sd/sqrt(R*V)
(all.rrcv = cbind(Mis.cv[,1:2],round(cbind(RRMi,RRMi.CIl, RRMi.CIu),2)))
all.rrcv[order(RRMi),]

# > all.cv[order(Micv),]
# V1   V2    Micv Micv.CIl Micv.CIu
# 13  3 1.00 0.11514  0.11230  0.11798
# 4   3 0.10 0.11518  0.11244  0.11792
# 1   3 0.01 0.11522  0.11251  0.11793
# 10  3 0.75 0.11534  0.11248  0.11820
# 15 10 1.00 0.11551  0.11281  0.11822
# 7   3 0.50 0.11559  0.11258  0.11860
# 14  6 1.00 0.11565  0.11287  0.11843
# 12 10 0.75 0.11569  0.11281  0.11857
# 5   6 0.10 0.11579  0.11294  0.11864
# 2   6 0.01 0.11585  0.11300  0.11870
# 11  6 0.75 0.11603  0.11309  0.11897
# 6  10 0.10 0.11644  0.11371  0.11917
# 8   6 0.50 0.11652  0.11385  0.11919
# 9  10 0.50 0.11656  0.11394  0.11918
# 3  10 0.01 0.11782  0.11495  0.12069

# Here, selecting any of the first 4 models would be alright
# size = 3 with decay = 1 might be too biased and 
# size = 3 with decay = 0.01 might be too variable
# So I select the simpler model between the remaining two
# i.e. size =3 and decay = 0.1
best.size = 10
best.decay = 0.01
nrounds = 10
# Train the final model 
y.1 <- as.matrix(train.y)
x.1.unscaled <- as.matrix(train.x) 
x.1 <- rescale(x.1.unscaled, x.1.unscaled)

## Restart nnet nrounds times to get best fit for each set of parameters 
Mi.final <- 1
#  check <- MSE.final
for(i in 1:nrounds){
  nn <- nnet(y=y.1, x=x.1, size=best.size, 
             decay=best.decay, maxit=2000, softmax=TRUE, trace=FALSE)
  Pi <- predict(nn, newdata=x.1, type="class")
  Mi <- mean(Pi != train.y.class)
  
  if(Mi < Mi.final){ 
    Mi.final <- Mi
    nn.final <- nn
  }
}
# Get training set error
pred.nn = predict(nn.final, newdata=x.1, type="class")
train.mis = mean(pred.nn != train.y.class)
#  0.1107525
# confusion matrix
table(train.y.class, pred.nn, dnn = c("Obs", "Pred"))

# Load test data set
data = read.csv("test.csv")
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

# Create a numerical version of variables for methods that need numbers
data$job <- as.numeric((data$job))
data$marital <- as.numeric((data$marital))
data$education <- as.numeric((data$education))
data$default <- as.numeric((data$default))
data$housing <- as.numeric((data$housing))
data$loan <- as.numeric((data$loan))
data$poutcome <- as.numeric((data$poutcome))

#For simplicity, rename data as "train.x" and "train.y"
test.x = data[,-16]
test.y.class = data[,16] 
test.y = class.ind(test.y.class)
# Rescale test set
y.2 <- as.matrix(test.y)
x.2.unscaled <- as.matrix(test.x) 
x.2 <- rescale(x.2.unscaled, x.1.unscaled)


# Get test set error
pred.nn = predict(nn.final, newdata=x.2, type="class")
test.mis = mean(pred.nn != test.y.class)
# 0.128306
# confusion matrix
table(test.y.class, pred.nn, dnn = c("Obs", "Pred"))
