function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize values
m = length(y); % number of training examples

J = 0;
grad = zeros(size(theta));

%========== cost J =============

%hypothesis vector (m x 1) = X(m x n) * theta (n x 1)
h = X * theta;
%errors vector (m x 1) = hypothesis vector(m x 1) - y(m x 1) vector
e = h - y;
%regularization parameter
%set first entry of theta vector to 0 because we dont regularize intercept
temp_theta = theta;
temp_theta(1) = 0;
regularization = (lambda / (2 * m))  * sum(temp_theta.^2);
%final cost function (1/2m x sum of squared error vector + regularization parameter)
J = 1/(2*m) * sum(e.^2) + regularization;

%========== gradient =============

%hypothesis 
hyp = X * theta;
%error vect
e = hyp - y;
%regularization (use temp theta because intercept is not to be regularized)
regularization = (lambda * temp_theta) / m;
%gradient
grad = 1/m * X'*e + regularization;
grad = grad(:);

end

%{
=========== test cases =========
X = [[1 1 1]' magic(3)];
y = [7 6 5]';
theta = [0.1 0.2 0.3 0.4]';
[J g] = linearRegCostFunction(X, y, theta, ?)

--- results based on value entered for ? (lambda)
--------------------------
lambda = 0  |   lambda = 7
--------------------------
J = 1.3533  |   J = 1.6917
g =         |   g = 
   -1.4000  |      -1.4000
   -8.7333  |      -8.2667
  -4.3333  |      -3.6333
  -7.9333  |      -7.0000

  }%