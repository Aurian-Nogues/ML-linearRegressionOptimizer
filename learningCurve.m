function [error_train, error_val] = ...
    learningCurve(X, y, Xval, yval, lambda)
%LEARNINGCURVE Generates the train and cross validation set errors needed 
%to plot a learning curve
%   [error_train, error_val] = ...
%       LEARNINGCURVE(X, y, Xval, yval, lambda) returns the train and
%       cross validation set errors for a learning curve. In particular, 
%       it returns two vectors of the same length - error_train and 
%       error_val. Then, error_train(i) contains the training error for
%       i examples (and similarly for error_val(i)).

% Number of training examples
m = size(X, 1);

% return these values
error_train = zeros(m, 1);
error_val   = zeros(m, 1);

for i = 1:m;
    % minimize error for Theta in given set
    theta_set = trainLinearReg(X(1:i, :),y(1:i), lambda);
    %calculate training error
    [J, grad] = linearRegCostFunction(X(1:i, :), y(1:i), theta_set, 0);
    error_train(i) = J;
    %calculate cross validation error
    [J, grad] = linearRegCostFunction(Xval, yval, theta_set, 0);
    error_val(i) = J;
end


% ============ test cases =========

%   X = [ones(5,1) reshape(-5:4,5,2)];
%   y = [-2:2]';
%   Xval=[X;X]/10;
%   yval=[y;y]/10;
%   [et ev] = learningCurve(X,y,Xval,yval,1)

%   et =

%      0.000000
%      0.031250
%      0.013333
%      0.005165
%      0.002268

%   ev =

%     3.0000e-002
%     5.3125e-003
%     6.0000e-004
%     9.2975e-005
%     2.2676e-005