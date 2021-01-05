set.seed(9386311)
# read in the data
data = read.csv("./archive/bank-additional-full.csv")
head(data)
summary(data)

# remove rows with unknown observations
library(dplyr)
data = data %>%
  filter(default != "unknown") %>%
  filter(marital != "unknown") %>%
  filter(housing != "unknown") %>%
  filter(loan != "unknown")

# drop following variables: Contact, Month,Day of week,Campaign, duration
data = data[,c(-8,-9,-10,-11,-12)]
head(data)
summary(data)

#x11()
# pairs(data)

# Split the data into Train & Test Sets
n.total = nrow(data)
n.test =  0.2*n.total
sample.ids = sample(n.total, n.test)
d.test = data[sample.ids,]
d.train = data[-sample.ids,]

# check the percentage positives in the splits
sum(data$y == "yes")/nrow(data)
sum(d.train$y == "yes")/nrow(d.train)
sum(d.test$y == "yes")/nrow(d.test)

source("Helper.R")
# Make folds
### Number of folds
K = 5
### Get folds for the Training set
n = nrow(d.train)
folds = get.folds(n, K)
write.csv(folds, 'folds.csv', row.names=F, col.names=F)

# Preparing the five social & economic variables in the Test set using the mean & SD of the Training Set
d.test[,c(11,12,13,14,15)] = scale.1(d.test[,c(11,12,13,14,15)], d.train[,c(11,12,13,14,15)])

# save the sets
write.csv(d.test, './test.csv', row.names=F, col.names=F)
write.csv(d.train, './train.csv', row.names=F, col.names=F)