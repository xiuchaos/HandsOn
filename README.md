# Hands-On Machine Learning with Scikit-Learn and TensorFlow
**Resources:** [book](https://www.amazon.com/Hands-Machine-Learning-Scikit-Learn-TensorFlow/dp/1491962291) and [code](https://github.com/ageron/handson-ml)
> start from July 02, 2018

## Part I The fundamentals of machine learning

### 1. the machine learning landscape
**Testing and validating** 
* generalization error - out of sample error
* train multiple models with various hyperparameters using training data, select model and hyper-parameter that perform best on the validation set, and estimate the generalization error on test data.
* avoid wasting too much training data in validation set, use cross-validation. then models are training on the full training set, and the generalized error is measured on the test set.
> hyper-parameter decides the amount of regularization to apply during learning, it must be set prior to training and remains constant during training.


### 2. End-to-End machine learning project
   * Looking the big picture
      * frame the problem
      * select a performance measure
         ...root mean square errot (RMSE), 68-95-99.7
         ...mean absolute error (MAE), l-l norm
         ...l
      * check the assumptions
      < the first question to ask you boss is what exactly is the business objective. how does the company expect to benefit fro this model? Second question to ask is what the current solution looks like. It will ofter give you a reference performance, as well as insights on how to solve the problem.
      
   
   * Dicover and visulize the data to gain insights
     * visualizing geographical data
     * looking for correlations
     * experimenting with attribute combinations
     
   * Preparing the data for machine learning algrothims
     * data cleaning
     * custom transformers
     * feature scaling
     * transformation pipelines
   
  * Select and train a model
   
  * Fine-Tune your model
      * grid search
      * randomized search
      * ensemble methods
      * analyze the best models and their errors
 
 * Launch, monitor and maintain your system
     

### 3. Classification
   * performance measures
      ** accuracy using cross validation
      ** confusion matrix
     precision and recall
     precison/recall tradeoff
     the roc curve
     
### 4. Training Models 
   * Linear regression
   * Gradient descent
   * Polynomial regression
   * Regularized linear models
   * Logistic regression
 
### 5. Support vector machine
   * Linear SVM classification
   * Nonlinear SVM classification
   * SVM regression
   * Under the hood
   
### 6. Decision Tree
   
### 7. Ensemble learning and random forests

### 8. Dimensionality Reduction

----
# Part II Neural Network and Deep Learning
### 9. tensorflow
### 10. Neural networks
### 11. Deep Neural networks
### 13. Convolutional Neural networks
### 14. Recurrent Neural networks
### 15. Autoencoder
### 16. Reinforcement learning
