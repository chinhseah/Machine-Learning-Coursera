function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% Part 1:
% a1 equals the X input matrix with a column of 1's added (bias units) as the first column.
a1 = [ones(m,1), X];
%z2 equals the product of a1 and Theta1
z2 = a1 * Theta1.'; % 5000 x 401 * 401 x 25 = 5000 x 25
%a2 is the result of passing z2 through g()
a2 = sigmoid(z2);
%Then add a column of bias units to a2 (as first column)
a2 = [ones(m,1), a2]; % 5000 x 26
%z3 (output) equals product of a2 and Theta2
z3 = a2 * Theta2.'; % 5000 x 26 * 26 x 10 = 5000 x 10
% a3 is the result of passing z3 through g()
a3 = sigmoid(z3);
% J(theta) = 1/m(sum-i-m[sum-k-K[-y*log(a3) - (1 - y)log(1-a3)]])
h1 = log(a3);
h2 = log(1-a3);
% Expand the 'y' output values into a matrix of single values 
eye_matrix = eye(num_labels);
y_matrix = eye_matrix(y,:);
J_unreg = 0;
for i=1:m
    J_unreg = J_unreg + h1(i,:)*y_matrix(i, :).' + h2(i,:)*(1-y_matrix(i, :).'); 
end;

J_unreg = -1/m * J_unreg;
% Regularize the cost (excluding theta columns for bias units)
% J_unreg + lambda/2m(sum-1-25[sum-1-400[theta1^2]+sum-1-10[sum-1-25[theta2^2])
theta1_squared = Theta1(:,2:end).^2;
theta2_squared = Theta2(:,2:end).^2;
reg = (lambda/(2*m)) * (sum(sum(theta1_squared)) + sum(sum(theta2_squared)));
J = J_unreg + reg;

% Part 2: Implement the backpropagation algorithm
delta3 = a3 - y_matrix; % 5000 x 10 
delta2 = delta3 * Theta2(:,2:end).*sigmoidGradient(z2); % 5000 x 10 * 10x25 = 5000 x 25 * 5000 x 25
% Delta2 = delta3 * a2'
% Delta1 = delta2 * a1'
Delta2 = a2.' * delta3; % a2 is 5000x25 and delta3 is 5000x10 = 25x10
Delta1 = delta2' * a1; % delta2 is 5000x25 and a1 is 5000x401 = 25x401 
     
Theta1_grad = (1/m)*Delta1; % should be 25x401
Delta2 = Delta2.'; % transpose 25x10
Theta2_grad = (1/m)*Delta2; % should be 10x25

% Part 3: Implement regularization with the cost function and gradients.
% In order to not regularize the first column
Theta1_zeros = [zeros(size(Theta1,1), 1) Theta1(:, 2:end)];
Theta2_zeros = [zeros(size(Theta2,1), 1) Theta2(:, 2:end)];
% add on to previous un-regularize thetas
Theta1_grad = Theta1_grad + (lambda/m)*Theta1_zeros;
Theta2_grad = Theta2_grad + (lambda/m)*Theta2_zeros;
% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
