function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%
h = X * theta; % X is 12x2, theta is 2x1, h is 12x1 
delta = h - y; % delta is 12x1
J = sum((delta.^2)) / (2 * m); % un-regularized cost
J = J + (lambda*sum(theta(2:end).^2)) / (2 * m); % regularized cost

grad = X.'*delta / m % X is 12x2, delta is 12x1 grad is 2x1 un-regularized
grad = grad + lambda*[0;theta(2:end)] / m;

% =========================================================================

grad = grad(:);

end
