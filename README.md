# Bank Marketing Campaign Classification
This is my Project for STAT 652. Here I analyze and train models using data collected by a Bank's Marketing Campaign to determine whether a client will subscribe to a term deposit or not. In addition, I discuss model performance metrics the marketing team cares about and suggest actions the marketing team might take.

## Introduction
Digital marketing campaigns are ubiquitous as organizations compete for their client's attention. In this project I explore the challenge of creating an effective marketing strategy in the context of banks. The data set is taken from UCI's Machine Learning Repository, you can find a detailed description [here](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing). Using the data from the previously carried out marketing campaign and client information I build models to predict whether a client will subscribe to a term deposit or not in a future marketing campaign. Ultimately, the goal is to enable the marketing team to optimize their resources to reach out to those clients who have the highest likelihood to need and subscribe to the term deposit.

## Data Set Description
- Source: Marketing campaign data from a Portuguese Bank (2008 to 2010)
- Size: 41188 records, 20 explanatory variables, and 1 binary response variable
- Skewed dataset
   - 11% positive responses
- 23% of total records have missing Values in at least one explanatory variable 
    
## Data Set Preprocessing
The original data set contains explanatory variables that falls into one of the following categories:
1. Client Demographic Data
2. Current Campaign Data
3. Previous Campaign Data
4. Social and Economic Attributes

### Dropping Explanatory Variables
I drop the explanatory variables in the Current Campaign Data category for two reasons. Firstly, data from a current campaign would be highly indicative of the outcome, and lead to data leakage. By using these explanatory variables I would be cheating. 
Secondly, data from a current marketing campaign along with it's outcome would not be available immediately. It would take a few weeks to know the outcome of the current marketing campaign in a real world setting.
Therefore for fairness and practicality, I drop the explanatory variables in the Current Campaign Data category.
### Dropping Records
For simplicity I decided to drop the records with any missing values. To rationalize this action, I checked that we still have enough records for training the models and that the proportion of positive samples is the same. 
In a future iteration of this project I will tackle the problem of missing values with imputation methods.

After the preprocessing steps above we now have a dataset with 31760 records and 15 explanatory variables.
### Train-Test Split
The data set is split into a training set and test set in a 80/20 ratio. The training set is used to find the best model and that model's best hyperparameters. The test set is used to evaluate the best model's predictive capability. 

### Standardize the five social and economic indicator variables
Since these variables have widely different range of values I standardized them to have a mean of zero and standard deviation of 1. I standardized the test set values using the mean and standard deviation of the training set to prevent any data leakage.

## Model Training Process
I will be evaluating the following models:
1. Logistic Regression (intended to serve as a baseline model)
2. Neural Networks
3. Random Forest
4. Naive Bayes

### Model Selection
To compare different model families and select the optimal hyperparameters I use cross validation on the training set as follows:
- Save out the 5 folds so it can be used for all models. (This will prevent introducing additional variance).
- The first time, take the 4 training folds.
- To tune a model, I perform another 5 fold cross validation at this stage to find the optimal hyperparameters.
- I select the hyperparameters corresponding to the lowest mean misclassification error across the 5 inner validation set folds. 
- Train my model with these hyperparameters on the 4 training folds.
- Use the validation fold to evaluate this model's performance.
- Repeat this process for all the models.

The average misclassifcation error rate over the 5 Cross Validation folds for different models,

| Model | Test Set Error % |Accuracy % |
| ------ | ------| ------ |
| Logistic Regression|0 |0|
| Neural Network |0.128306 |0.871694|
| Random Forest |0 |0 |
| Naive Bayes |0 |0 |

This is the box plots of the misclassification errors from 5 folds of Cross Validation for thr four models.
![boxplot of all models](https://github.com/missrhea/Bank-Marketing-Campaign-Classification/blob/main/images/boxplot%20of%20all%20models.png)

To generate the box plots of relative misclassification errors, for each fold I divide the misclassification error of each model by the minimum misclassification error in the same fold. This makes it simple to interpret how much worse / off models are relative to the best one.
![relative boxplot of all models](https://github.com/missrhea/Bank-Marketing-Campaign-Classification/blob/main/images/relative%20boxplot%20of%20all%20models.png)

For this problem it is helpful to take a look at the confusion matrix for two reason,
1. The data set is highly imbalanced (only ~12% positive samples).
2. A marketing team has limited resources and would like to minimize the number of clients they reach out to, i.e. minimize reaching out to clients who will not subscribe to a term deposit. This corresponds to minimizing the False Positives. 

![confusion matrix all](https://github.com/missrhea/Bank-Marketing-Campaign-Classification/blob/main/images/confusion%20matrix%20all.png)

### Final Model Training
- Finally select the model with the lowest mean misclassification error across the 5 outer validation set folds.
- Tune this selected model using a 5 fold CV (for convinience I use the 5 folds I saved). 
- Select the best hyperparameters corresponding to the lowest mean misclassification error rate.
- Train the model on the entire training set.

These are the box plots made using misclassification errors from the fold of Cross Validation done during Tuning with different hyperparameter settings.

![nn hyperparameters boxplot](https://github.com/missrhea/Bank-Marketing-Campaign-Classification/blob/main/images/nn%20hyperparameters%20boxplot.png)

Here is the box plot using relative misclassification errors.

![relative nn hyperparameters boxplot](https://github.com/missrhea/Bank-Marketing-Campaign-Classification/blob/main/images/relative%20nn%20hyperparameters%20boxplot.png)

### Evaluating the Model on the Test Set
##### Optimal Hyperparameters
* size = 10
* decay = 0.01

##### Test Set Misclassification Error Rate

Here is the misclassification error rate on the test data set.

| Model | Test Set Error % |Accuracy % |
| ------ | ------| ------ |
| Neural Network |0.128306 |0.871694|

This very high Accuracy on the test set is a misleading metric to evaluate model performance. This is especially common with data sets that are highly imbalanced like this one. To get a better picture of the model performance we look at the Confusion Matrix next.

##### Test Set Confusion Matrix
Accuracy is not the best evaluation metric for unbalanced data sets like this one. Furthermore, the cost associated with False Positives and False Negatives is different. In a marketing setting, we care more about minimizing False Positives than minimizing the False Negatives.

In other words, if a client is highly likely to get the term deposit but our model predicted that the client would not get the term deposit that is a False Negative. If we missed a client who will get a term deposit anyway that is alright.

On the other hand, if a client is highly unlikely to get the term deposit but our model predicted that the client will get the term deposit that is a False Positive. Missing a client we could have converted with a marketing offer will lead to a loss.

Thus, the marketing team has a higher tolerance for False Negatives compared to False Positives. The Confusion Matrix gives us a way to visualize this tradeoff. Precision is a metric obtained from the Confusion Matrix, defined as

Precision = TP/(TP+FP)

Using precision in this context is useful since it provides an idea of the cost incurred due to the False Positives.

This is the confusion matrix of the predictions on the test set.
![NN Test set confusion matrix](https://github.com/missrhea/Bank-Marketing-Campaign-Classification/blob/main/images/NN%20Test%20set%20confusion%20matrix.png)

Here we see that the Neural Network model is not at all useful. It deceptively achieves a low misclassification error rate by predicting "no" for every test set instance. This is not any better than guessing. 

## Pivot
### Motivation
To address the bias in the Neural Network model, I look at the confusion matrix for different models from the 5 fold cross validation done during the model selection phase.

![confusion matrix all](https://github.com/missrhea/Bank-Marketing-Campaign-Classification/blob/main/images/confusion%20matrix%20all.png)

Keeping in mind the trade off between the bias and variance among the models, I decide to pivot to using the Random Forest model.

### Final Model Training
This is the box plot of misclassification errors from the cross validation using different values for mtry & nodesize hyperparameters.
![RF relative boxplot](https://github.com/missrhea/Bank-Marketing-Campaign-Classification/blob/main/images/RF%20relative%20boxplot.png)

I decided to fine tune the hyperparameters using a finer grid.
![RF relative boxplot Fine Tuned](https://github.com/missrhea/Bank-Marketing-Campaign-Classification/blob/main/images/RF%20relative%20boxplot%20Fine%20Tuned.png)

### Evaluating the Model on the Test Set
##### Optimal Hyperparameters
* mtry = 3
* nodesize = 25

##### Test Set Misclassification Error Rate

| Model | Test Set Error % |Accuracy % |
| ------ | ------| ------ |
| Random Forest | 0.1116008 |0.8883992|

##### Test Set Confusion Matrix


## Suggestions to the Marketing Team
1. By personalized marketing to a percentage of clients ranked by the model you can convert more clients than if you would have reached out to them randomly.
![cph chart](https://github.com/missrhea/Bank-Marketing-Campaign-Classification/blob/main/images/cph%20chart.png)

2. Consult with experts in economics since the XYZ social-economic explanatory variables greatly influences the outcome.
![RF var importance plot](https://github.com/missrhea/Bank-Marketing-Campaign-Classification/blob/main/images/RF%20var%20importance%20plot.png)




## Future Work
- Handling the missing values with imputation methods.
- Use AUC instead of misclassification error (i.e. accuracy)

## References
- On normalization and standardization for feature transformation: https://towardsai.net/p/data-science/how-when-and-why-should-you-normalize-standardize-rescale-your-data-3f083def38ff
- Evaluation Criteria for classification and the use of measures relevant in Marketing: https://www.mimuw.edu.pl/~son/datamining/DM/eval-train-test-xval.pdf
- ROC Curves: https://www.hackernoon.com/making-sense-of-real-world-data-roc-curves-and-when-to-use-them-90a17e6d1db
 
