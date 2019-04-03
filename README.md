# ML-linearRegressionOptimizer
Matlab/Octave program to optimise a linear regression (polynomial and lambda)

data.mat is a dataset containing data about a dam. X is the water level and y the water flowing out of the dam.
featureNormalize.m - Feature normalization function
fmincg.m - Function minimization routine
plotFit.m - Plot a polynomial t
trainLinearReg.m - Trains linear regression using cost function
linearRegCostFunction.m - Regularized linear regression cost function
polyFeatures.m - Maps data into polynomial feature space
validationCurve.m - Generates training and cross validation error vectors depending on lambda
learningcurve.m - Generates training and cross validation error vectors for size of training set
optimizer.m - Scrit that runs all the functions to find good parameters for the algorithm




