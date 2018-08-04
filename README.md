## Hands-On Machine Learning with Scikit-Learn and TensorFlow
**Resources:** [book](https://www.amazon.com/Hands-Machine-Learning-Scikit-Learn-TensorFlow/dp/1491962291) and [code](https://github.com/ageron/handson-ml)
> start from July 02, 2018

## 1.The machine learning landscape
> Machine Learning is the field of study that give computers the ability to learn without being explicityly programmed.

> A computer program is said to learn formew experience E with respect to some task T and some performance measure P, if its performance on T, as measured by P, improves with experience E.

> Learning means getting better at some tasks, given some performance measure.


**Testing and validating** 
* generalization error - out of sample error
* train multiple models with various hyperparameters using training data, select model and hyper-parameter that perform best on the validation set, and estimate the generalization error on test data.
* avoid wasting too much training data in validation set, use cross-validation. then models are training on the full training set, and the generalized error is measured on the test set.
    > hyper-parameter decides the amount of regularization to apply during learning, it must be set prior to training and remains constant during training.


## 2. End-to-End machine learning project

**Looking the big picture**
* frame the problem
* select a performance measure
  + root mean square errot (RMSE), 68-95-99.7
  + mean absolute error (MAE), l-l norm, 12 more sensitive to noise than l1
  + l-k norm, the higher the norm index, the more it focuses on large values and neglect small ones
* check the assumptions
   > the first question to ask you boss is what exactly is the business objective. how does the company expect to benefit from this model? Second question to ask is what the current solution looks like. It will ofter give you a reference performance, as well as insights on how to solve the problem.

**Get the data**
* Create a TestSet

**Dicover and visulize the data to gain insights**
* visualizing geographical data
* looking for correlations
* experimenting with attribute combinations 
  + identified a few data quirks that may need to clean up before feding the data to a ML algorithm; 
  + find interesting correaltions between attributes; 
  + some attributes have a tail-heavy distribution, you may want to transfrom them;
  + may helpful to try out various attribute combinations.

**Preparing the data for machine learning algorithms**
> write functions to transform the data,gradually build a library of transformations functions for reuse in future projects. 
* data cleaning
* custom transformers
* feature scaling
* transformation pipelines 

**Select and train a model**

**Fine-Tune your model**
* grid search
* randomized search
* ensemble methods
* analyze the best models and their errors

**Launch, monitor and maintain your system**
     

##  3.Classification
**performance measures**
* accuracy using cross validation
* confusion matrix
* precision and recall
* precison/recall tradeoff
* the roc curve
     
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
