function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
% For both C and sigma, we suggest trying values in multiplicative steps 
% (e.g. 0:01; 0:03; 0:1; 0:3; 1; 3; 10; 30).
Ctest = [0.01 0.03 0.1 0.3 1 3 10 30]; % 1x16
sigmaTest = [0.01 0.03 0.1 0.3 1 3 10 30]; % 1x16
% store prediction errors in 16x16 row = C, column = sigma
prediction_errors = zeros(length(Ctest),length(sigmaTest)); 
% loop over each combination of C and sigma test values
for cIndex = 1:length(Ctest)
    C = Ctest(cIndex);
    for sIndex = 1:length(sigmaTest)
        sigma = sigmaTest(sIndex);
        model = svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
        % Xval is 200x2 yval is 200x1
        predictions = svmPredict(model,Xval); %predictions pred is a m x 1 column of predictions of {0, 1} values.
        prediction_errors(cIndex,sIndex) = mean(double(predictions ~= yval));
    end
end
% need to find minimum of prediction_errors
[M,I] = min(prediction_errors(:));
[I_row, I_col] = ind2sub(size(prediction_errors),I);
C = Ctest(I_row);
sigma = sigmaTest(I_col);
% =========================================================================

end
