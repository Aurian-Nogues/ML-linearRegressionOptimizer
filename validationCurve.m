function [lambda_vec, error_train, error_val] = ...
    validationCurve(X, y, Xval, yval)
%VALIDATIONCURVE Generate the train and validation errors needed to
%plot a validation curve that we can use to select lambda
%   [lambda_vec, error_train, error_val] = ...
%       VALIDATIONCURVE(X, y, Xval, yval) returns the train
%       and validation errors (in error_train, error_val)
%       for different values of lambda. You are given the training set (X,
%       y) and validation set (Xval, yval).
%

% Selected values of lambda
lambda_vec = [0 0.001 0.003 0.01 0.03 0.1 0.3 1 3 10]';

% variables to be returned
error_train = zeros(length(lambda_vec), 1);
error_val = zeros(length(lambda_vec), 1);


for i = 1:length(lambda_vec)
    %select lambda and train the algo to get theta
    lambda = lambda_vec(i);
    theta = trainLinearReg(X, y, lambda);
    %store training cost
    [J, grad] = linearRegCostFunction(X, y, theta, 0);
    lambda_vec(i) = lambda;
    error_train(i) = J;
    %compute cross validation cost and store it
    [J, grad] = linearRegCostFunction(Xval, yval, theta, 0);
    error_val(i) = J;
end

end



% ============================== test case ==========================

%{
X = [1 2 ; 1 3 ; 1 4 ; 1 5];
y = [7 6 5 4]';
Xval = [1 7 ; 1 -2];
yval = [2 12]';
[lambda_vec, error_train, error_val] = validationCurve(X,y,Xval,yval )

% results:
lambda_vec =
    0.00000
    0.00100
    0.00300
    0.01000
    0.03000
    0.10000
    0.30000
    1.00000
    3.00000
   10.00000

error_train =

   0.00000
   0.00000
   0.00000
   0.00000
   0.00002
   0.00024
   0.00200
   0.01736
   0.08789
   0.27778

error_val =

   0.25000
   0.25055
   0.25165
   0.25553
   0.26678
   0.30801
   0.43970
   1.00347
   2.77539
   6.80556

   }%