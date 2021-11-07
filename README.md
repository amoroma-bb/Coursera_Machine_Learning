# Coursera_Machine_Learning
## Week 1
### Introduction
- Supervised Learning
  - Regression -> Predict continuous output
    - Example: Given data about the size of houses on the real estate market, try to predict the price.
  - Classification -> Discrete value of output
    - Example: Given a patient with a tumor, predict whether the tumor is malignant or benign
- Unsupervised Learning
  - Unsupervised learning allows us to approach problems with little or no idea what our results should look like.
  - We can derive structure from data where we don't know necessarily know the effect of the variables. 
  - Clustering
### Model and Cost Function
  - Cost Function
### Parameter Learning
  - Gradient Descent
### Linear Algebra
  - Matrices and Vectors
  - Addition and Scalar Multiplication
  - Matrix Multiplication
  - Inverse and Transpose

## Week 2
### Environment Setup Instruction
  - Octave or MATLAB
  - ![equation](https://latex.codecogs.com/svg.image?h(x)&space;=\theta&space;_{0}*x_{0}&space;&plus;\theta&space;_{1}*x_{1}&space;&plus;&space;\theta&space;_{2}*x_{2}&space;&plus;&space;...&space;&plus;&space;\theta&space;_{n}*x_{n})
### Multivariate Linear Regressioon
  - Multiple Features
  - Gradient Descent For Multiple Variables
    - Normalization
  - Features and Polynmial Regression
### Coputing Parameters Analytically
  - Normal Equation
    - Don't need α
    - Don't need to iterate
    - Need to compute ![equation](https://latex.codecogs.com/svg.image?(X^{T}*X)^{-1})
    - Slow if n is very large (eg. n > 10000)
  - Normal Equation Noninvertibility

## Week 3
### Classification and Representation
  - Classification
    - Logistic Regression
  - Hypothesis Representation
    - sigmoid function
    - ![equation](https://latex.codecogs.com/svg.image?h_{\theta}(X)&space;=&space;g(\theta^TX))
    - ![equation](https://latex.codecogs.com/svg.image?g(z)%20=%201/(1&plus;e%5E%7B-z%7D)) 
  - Decision Boundary

### Logistic Regression Model
  - Cost Function
    - ![euqation](https://latex.codecogs.com/svg.image?Cost(h_%7B%5Ctheta%7D(x),y)%20=%5Cbegin%7Bcases%7D%20-log(h_%7B%5Ctheta%7D(x))%20&%20%5Ctext%7B%20if%20%7D%20y=1%20%5C%5C%20-log(1-h_%7B%5Ctheta%7D(x))&%20%5Ctext%7B%20if%20%7D%20y=0%20%5Cend%7Bcases%7D)
  - Simplified Cost Function and Gradient Descent
    - ![equation](https://latex.codecogs.com/svg.image?Cost(h_{\theta}(x),y)&space;=&space;-ylog(h_{\theta}(x))-(1-y)log(1-h_{\theta}(x)))
  - Advanced Optimization 
    1. Gradient descent
    2. Conjugate gradient
    3. BFGS
    4. L-BFGS

    ` 2-4 No need to manually pick α. Often faster than gradient descent.
    More complex!`

    ```MATLAB
    function [jVal, gradient] = costFunction(theta)
      jVal = [...code to compute J(theta)...];
      gradient = [...code to compute derivative of J(theta)...];
    end
    ```

    ```MATLAB
    options = optimset('GradObj', 'on', 'MaxIter', 100);
    initialTheta = zeros(2,1);
    [optTheta, functionVal, exitFlag] = fminunc(@costFunction, initialTheta, options);
    ```
### Multiclass Classification
  - Multiclass Classification: One vs all
  
### Solving the Probelm of Overfitting
  - The Problem of Overfitting
    1. Reduce number of features
        - Manually select which features to keep
        - Model selection algorithm
    2. Regularization
        - Keep all the features, but reduce magnitude / values of parameters
        - Works well when we have a lot of features, each of which contributes a bit to predicting y
  - Cost Function
    1. $min_{\theta}\frac{1}{2m}\sum_{i=1}^{m}(h_{\theta}(x^{(i)})-y^{(i)})^{2}+\lambda\sum_{j=1}^{n}\theta_{j}^{2}$
  
  - Regularized Linear Regression
    $
    \begin{align*} & \text{Repeat}\ \lbrace \newline & \ \ \ \ \theta_0 := \theta_0 - \alpha\ \frac{1}{m}\ \sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)})x_0^{(i)} \newline & \ \ \ \ \theta_j := \theta_j - \alpha\ \left[ \left( \frac{1}{m}\ \sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)})x_j^{(i)} \right) + \frac{\lambda}{m}\theta_j \right] &\ \ \ \ \ \ \ \ \ \ j \in \lbrace 1,2...n\rbrace\newline & \rbrace \end{align*}
    $

    $
    \newline \newline
    \begin{align*}& \theta = \left( X^TX + \lambda \cdot L \right)^{-1} X^Ty \newline& \text{where}\ \ L = \begin{bmatrix} 0 & & & & \newline & 1 & & & \newline & & 1 & & \newline & & & \ddots & \newline & & & & 1 \newline\end{bmatrix}\end{align*}
    $
  - Regularized Logistic Regression