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
    ``` MATLAB
    function [jVal, gradient] = costFunction(theta)
      jVal = [code to compute J(theta)];

      gradient(1) = [code to compute derivative of theta0];

      gradient(2) = [code to compute derivative of theta1];

      .
      .
      .

      gradient(n+1) = [code to compute derivative of thetan];
    ```
## Week 4
### Motivations
  - Non-linear Hypotheses
  - Neurons and the Brain

### Neural Networks
  - Model Representation 1
   
  $
  [x_{0} \ x_{1} \ x_{2}] -> [a_{1}^{2} \ a_{2}^{2} \ a_{3}^{2}] -> h_{\theta}(x)
  $ 

  $
  \begin{align*} a_1^{(2)} = g(\Theta_{10}^{(1)}x_0 + \Theta_{11}^{(1)}x_1 + \Theta_{12}^{(1)}x_2 + \Theta_{13}^{(1)}x_3) \newline a_2^{(2)} = g(\Theta_{20}^{(1)}x_0 + \Theta_{21}^{(1)}x_1 + \Theta_{22}^{(1)}x_2 + \Theta_{23}^{(1)}x_3) \newline a_3^{(2)} = g(\Theta_{30}^{(1)}x_0 + \Theta_{31}^{(1)}x_1 + \Theta_{32}^{(1)}x_2 + \Theta_{33}^{(1)}x_3) \newline h_\Theta(x) = a_1^{(3)} = g(\Theta_{10}^{(2)}a_0^{(2)} + \Theta_{11}^{(2)}a_1^{(2)} + \Theta_{12}^{(2)}a_2^{(2)} + \Theta_{13}^{(2)}a_3^{(2)}) \newline \end{align*}
  $
  
  - Model Representation 2
  $
  \begin{align*}a_1^{(2)} = g(z_1^{(2)}) \newline a_2^{(2)} = g(z_2^{(2)}) \newline a_3^{(2)} = g(z_3^{(2)}) \newline \end{align*}
  $
  
  $\begin{align*}x = \begin{bmatrix}x_0 \newline x_1 \newline\cdots \newline x_n\end{bmatrix} &z^{(j)} = \begin{bmatrix}z_1^{(j)} \newline z_2^{(j)} \newline\cdots \newline z_n^{(j)}\end{bmatrix}\end{align*}
  $

  $
  z^{(j)} = \Theta^{(j-1)}a^{(j-1)}
  \newline 
  \newline
  z^{(j+1)} = \Theta^{(j)}a^{(j)}
  $
  We get the final z vector by multiplying the next matrixs after $\Theta^{(j-1)}$ with the values of all the activation nodes we just got. The last theta matrix $\Theta^{j}$ will have only one row which is multiplied by one column $a^{(j)}$ so that our result is a single number.

  $
  h_{\Theta}(x) = a^{(j+1)} = g(z^{(j+1)})
  $
### Applications
  - Examples and Intuitions
  - Multiclass Classification

## Week 5
### Cost Function and Backpropagation
  - L = total no. of layers in network
  - $s_{l}$ = no. units (not counting bias unit) in layer l
  - K = output units
  - Neural networks cost function 
  $\begin{gather*} J(\Theta) = - \frac{1}{m} \sum_{i=1}^m \sum_{k=1}^K \left[y^{(i)}_k \log ((h_\Theta (x^{(i)}))_k) + (1 - y^{(i)}_k)\log (1 - (h_\Theta(x^{(i)}))_k)\right] + \frac{\lambda}{2m}\sum_{l=1}^{L-1} \sum_{i=1}^{s_l} \sum_{j=1}^{s_{l+1}} ( \Theta_{j,i}^{(l)})^2\end{gather*}
  $
  - Backpropagation Algorithm
  - Backpropagation Intuition

### Backpropagation in Practice
  - With nerual networks, we are working with sets of matrices
    $\begin{align*} \Theta^{(1)}, \Theta^{(2)}, \Theta^{(3)}, \dots \newline D^{(1)}, D^{(2)}, D^{(3)}, \dots \end{align*}
    $

    In order to use optimizing functions such as "fminunc()", we will want to "unroll" all the elements and put them into one long vector:

    ``` MATLAB
    thetaVector = [ Theta1(:); Theta2(:); Theta3(:); ]
    deltaVector = [ D1(:); D2(:); D3(:) ]
    ```

    Get back our original matrices from the "unrolled" versions:
    ```MATLAB
    Theta1 = reshape(thetaVector(1:110),10,11)
    Theta2 = reshape(thetaVector(111:220),10,11)
    Theta3 = reshape(thetaVector(221:231),1,11)
    ```
  
  - Gradient Cheching
  Once verified once that backpropagation algorithm is correct, then close compute gradApprox. Becasue gradApprox is slow.
  ```MATLAB
  epsilon = 1e-4;
  for i = 1:n,
    thetaPlus = theta;
    thetaPlus(i) += epsilon;
    thetaMinus = theta;
    thetaMinus(i) -= epsilon;
    gradApprox(i) = (J(thetaPlus) - J(thetaMinus))/(2*epsilon)
  end;
  ```
  - Random Initialization
    
    Random initialization: Symmetry breaking
    ```MATLAB
    If the dimensions of Theta1 is 10x11, Theta2 is 10x11 and Theta3 is 1x11.

    Theta1 = rand(10,11) * (2 * INIT_EPSILON) - INIT_EPSILON;
    Theta2 = rand(10,11) * (2 * INIT_EPSILON) - INIT_EPSILON;
    Theta3 = rand(1,11) * (2 * INIT_EPSILON) - INIT_EPSILON;

    ```
  
  - Put it Together
    Training a neural network

    No. of input units: Dimension of features

    No. of output units: Number of classes

    Reasonable default: 1 hidden layer, or if > 1 hidden layer, have same no. of hidden units in every layer (usually the more the better)

    1. Randomly initialize weights
    2. Implement of forward propagation to get $h_{\Theta}(x^{(i)})$ for any $x^{(i)}$
    3. Implement code to compute cost function $J(\Theta)$
    4. Implement backpropagation to compute partial derivatives
    5. Use gradient cheching
    6. Use gradient descent or advanced optimization method with backpropagation to try to minimize J as a function of parameters $\Theta$

## Week 6

### Evaluating a Learning Algorithm
  - Evaluating a Hypothesis
  - Bias vs. Variance
  - Learning Curves

### Building a Spam Classifier
  - Prioritzing What to Work on
  - Error Analysis 
    1. Start with a simple algorithm that you can impelement quickly
    2. Plot learning curves to decide if more data, more features, etc. are likely to help
    3. Error analysis: Manually examine the examples that your algorithm made errors on.

### Handling Skewed Data
  - Error Metrics for Skewed
  - Trading Off Precision and Recall

### Using Large Data Sets
  - Data For Machine Learning

## Week 7
### Large Margin Classification
  - Optimization Objective
  - Large Margin Intuition
  - Mathematics Behind Large Margin Classification

### Kernels

### SVMs in Practice

## Week 8
### Clustering
  - Unsupervised Learning
  - K-Means Algorithm
  - Optimization Objective
    1. $c^{(i)}$ = index of cluster to which example $x^{(i)}$ is currently assigned
    2. $\mu_{k}$ = cluster centroid k
    3. $\mu_{c^{(i)}}$ = cluster centroid of cluster to which example has been assigned
  - Random Initialization
  - Choosing the Number of Clusters

### Motivation
  - Data Compression
  - Visualization

### Principla Component Analysis

### Applying PCA
  - Reconstruction from Compressed Representation
  - Choosing the Number of Principal Components
    - 99 % of variance is retained
  - Advice for Applying PCA
    - Design of ML system
      1. Get training set
      2. RUn PCA to reduce x in dimension to get z
      3. Train logistic regression on {z,y}
      4. Test on test set: Map x to z
    
## Week 9
### Density Estimation

### Building an Anomaly Detection System

### Multivariate Gaussian Distribution

### Predicting Moive Ratings

### Collaborative Filtering

### Low Rank Matrix Factorization