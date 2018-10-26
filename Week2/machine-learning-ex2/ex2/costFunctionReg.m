function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
% calculate cost J
H = sigmoid(X * theta); % gives m x 1 matrix
E = y.'*log(H) + (1-y).'*log(1 - H);
theta(1)=0;
tsquares = theta.'*theta;
J = (-1/m * sum(E)) + (lambda/(2*m) * sum(tsquares));
% calculate grad - derivative cost
for i = 1 : m
	grad = grad + (H(i) - y(i)) * X(i,:)';
end
grad = 1/m*grad + (lambda/m)*theta;





% =============================================================

end
