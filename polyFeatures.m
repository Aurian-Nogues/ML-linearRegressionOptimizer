function [X_poly] = polyFeatures(X, p)
%POLYFEATURES Maps X (1D vector) into the p-th power
%   [X_poly] = POLYFEATURES(X, p) takes a data matrix X (size m x 1) and
%   maps each example into its polynomial features where
%   X_poly(i, :) = [X(i) X(i).^2 X(i).^3 ...  X(i).^p];
%

X_poly = zeros(numel(X), p);

X_poly = X;
if p >1;
    for i = 2:p;
        X_poly = [X_poly, X.^i ];
    end
end

end


%============= test cases =========

%   polyFeatures([1:7]',4)
%   ans =

%      1      1      1      1
%      2      4      8     16
%      3      9     27     81
%      4     16     64    256
%      5     25    125    625
%      6     36    216   1296
%      7     49    343   2401

