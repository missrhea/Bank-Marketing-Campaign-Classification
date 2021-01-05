#########################################################################
# Some helper functions 
#########################################################################

### Let's define a function for constructing CV folds
get.folds = function(n, K) {
  ### Get the appropriate number of fold labels
  n.fold = ceiling(n / K) # Number of observations per fold (rounded up)
  fold.ids.raw = rep(1:K, times = n.fold) # Generate extra labels
  fold.ids = fold.ids.raw[1:n] # Keep only the correct number of labels
  ### Shuffle the fold labels
  folds.rand = fold.ids[sample.int(n)]
  return(folds.rand)
}

### Rescale the explanatories for the NN models
rescale <- function(x1,x2){
  for(col in 1:ncol(x1)){
    a <- min(x2[,col])
    b <- max(x2[,col])
    x1[,col] <- (x1[,col]-a)/(b-a)
  }
  x1
}

### For discriminant analysis, it's best to scale predictors
### to have mean 0 and SD 1 (this makes the results easier to 
### interpret). We can do this using using the following function.

### Rescale x1 using the means and SDs of x2
scale.1 <- function(x1,x2){
  for(col in 1:ncol(x1)){
    a <- mean(x2[,col])
    b <- sd(x2[,col])
    x1[,col] <- (x1[,col]-a)/b
  }
  x1
}