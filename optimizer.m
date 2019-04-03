%% Initialization
clear ; close all; clc

%======== load and visualize data ======
fprintf('Loading and Visualizing Data ...\n')

% You will have X, y, Xval, yval, Xtest, ytest in your environment
load ('data.mat');
% m = Number of examples
m = size(X, 1);
% Plot training data
plot(X, y, 'rx', 'MarkerSize', 10, 'LineWidth', 1.5);
xlabel('Change in water level (x)');
ylabel('Water flowing out of the dam (y)');

input('press enter');

%======== train algo with no adjustments and vizualize fit ======
fprintf('Train algo without adjustments... \n');

%  Train linear regression with lambda = 0
lambda = 0;
[theta] = trainLinearReg([ones(m, 1) X], y, lambda);

fprintf('Visualising fit with no adjustments... \n');

%  Plot fit over the data
plot(X, y, 'rx', 'MarkerSize', 10, 'LineWidth', 1.5);
xlabel('Change in water level (x)');
ylabel('Water flowing out of the dam (y)');
hold on;
plot(X, [ones(m, 1) X]*theta, '--', 'LineWidth', 2)
hold off;

input('press enter');
%======== learning curve for linear regression =========
%draw learning curve to determine if we have biais/variance in model

fprintf('Calculating and visualizing learning curve... \n');

% set lambda to 0 not to pollute results with random parameters
lambda = 0;
[error_train, error_val] = ...
    learningCurve([ones(m, 1) X], y, ...
                  [ones(size(Xval, 1), 1) Xval], yval, ...
                  lambda);

plot(1:m, error_train, 1:m, error_val);
title('Learning curve for linear regression')
legend('Train', 'Cross Validation')
xlabel('Number of training examples')
ylabel('Error')
axis([0 13 0 150])

fprintf('If curves plateau quickly and stabilise at a high cost close to each other model has high biais (underfit): \n');
fprintf('-get more features \n-add polynomial features \n-decrease lambda \n');
fprintf('If curves plateau far from each other (i.e. training doesnt generalise well) model has high variance (overfit)\n')
fprintf('-add more training data \n-use less features \n-increase lambda \n')
input('press enter');

%===== feature mapping for polynomial regression ==========
%model has high biais, let's add some polynomial features to increase fit

fprintf('Adding polynomial features to data... \n')

% choose degree of polynomial to test
p = 8;
fprintf('Chosen polynomial degree: %f \n' , p);

fprintf('Mapping polynomial on cross validation and test sets and normalizing... \n')

%map X onto polynomial features and normalize
X_poly = polyFeatures(X, p);
[X_poly, mu, sigma] = featureNormalize(X_poly);  % Normalize
X_poly = [ones(m, 1), X_poly];                   % Add Ones

% Map X_poly_test and normalize (using mu and sigma)
X_poly_test = polyFeatures(Xtest, p);
X_poly_test = bsxfun(@minus, X_poly_test, mu);
X_poly_test = bsxfun(@rdivide, X_poly_test, sigma);
X_poly_test = [ones(size(X_poly_test, 1), 1), X_poly_test];         % Add Ones

% Map X_poly_val and normalize (using mu and sigma)
X_poly_val = polyFeatures(Xval, p);
X_poly_val = bsxfun(@minus, X_poly_val, mu);
X_poly_val = bsxfun(@rdivide, X_poly_val, sigma);
X_poly_val = [ones(size(X_poly_val, 1), 1), X_poly_val];           % Add Ones

%======= learning curve for polynomial regression ========

fprintf("Training polynomial model and visualising learning curve... \n")

lambda = 1;
[theta] = trainLinearReg(X_poly, y, lambda);

% Plot training data and fit
figure(1);
plot(X, y, 'rx', 'MarkerSize', 10, 'LineWidth', 1.5);
plotFit(min(X), max(X), mu, sigma, theta, p);
xlabel('Change in water level (x)');
ylabel('Water flowing out of the dam (y)');
title (sprintf('Polynomial Regression Fit (lambda = %f)', lambda));

figure(2);
[error_train, error_val] = ...
    learningCurve(X_poly, y, X_poly_val, yval, lambda);
plot(1:m, error_train, 1:m, error_val);

title(sprintf('Polynomial Regression Learning Curve (lambda = %f)', lambda));
xlabel('Number of training examples')
ylabel('Error')
axis([0 13 0 100])
legend('Train', 'Cross Validation')

fprintf('Polynomial Regression (lambda = %f)\n\n', lambda);
fprintf('# Training Examples\tTrain Error\tCross Validation Error\n');
for i = 1:m
    fprintf('  \t%d\t\t%f\t%f\n', i, error_train(i), error_val(i));
end

input('press enter');

%====== validation curve for selecting lambda ======== 
printf('Training algo for different lambdas and visualizing data...\n')

[lambda_vec, error_train, error_val] = ...
    validationCurve(X_poly, y, X_poly_val, yval);

close all;

plot(lambda_vec, error_train, lambda_vec, error_val);
legend('Train', 'Cross Validation');
xlabel('lambda');
ylabel('Error');

input('press enter');

%========= compute error on test set ============ 

fprintf('Calculating cost of using Theta and Lambda on test set \n');
%select best lambda from lambda validation curve
lambda = 3;
%train algo on training set
[theta] = trainLinearReg(X_poly, y, lambda);
fprintf('Calculating cost on test set... \n')
%calculate cost on test set
[J, grad] = linearRegCostFunction(X_poly_test, ytest, theta, lambda)
fprintf('Cost on training set is %f', J);



