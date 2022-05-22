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
input = [ones(m, 1), X];

z2 = input * Theta1';
a2 = sigmoid(z2);
a2 = [ones(m, 1), a2];

z3 = a2 * Theta2';
a3 = sigmoid(z3);

%[num, output] = max(z2, [], 2);

y_hot = zeros(m, size(a3, 2));

for i = 1:size(y, 1)
  num = y(i, 1);
  y_hot(i, num) += 1;
endfor

reg = lambda/(2*m) * (sum(sum(Theta1(:,2:end).^2)) + sum(sum(Theta2(:,2:end).^2)));

J = 1/m * sum(sum((-y_hot .* log(a3)) - (1 - y_hot) .* log(1 - a3))) + reg;

delta_3 = (a3 - y_hot);
#printf('SIZE OF DELTA3: %i x %i \n', size(delta_3, 1), size(delta_3, 2))
delta_2 = delta_3 * Theta2 .* [ones(size(z2, 1), 1), sigmoidGradient(z2)];
delta_2 = delta_2(:, 2:end);
#printf('SIZE OF DELTA2: %i x %i \n', size(delta_2, 1), size(delta_2, 2))

Theta1_grad(:, 1) = 1/m * (delta_2' * input(:, 1));
Theta1_grad(:, 2:end) = 1/m * (delta_2' * input(:, 2:end)) + lambda/m * Theta1(:, 2:end);
Theta2_grad(:, 1) = 1/m * (delta_3' * a2(:, 1));
Theta2_grad(:, 2:end) = 1/m * (delta_3' * a2(:, 2:end)) + lambda/m * Theta2(:, 2:end);

printf('SIZE OF THETA1_GRAD: %i x %i \n', size(Theta1_grad, 1), size(Theta1_grad, 2))
printf('SIZE OF THETA2_GRAD: %i x %i \n', size(Theta2_grad, 1), size(Theta2_grad, 2))
% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];
#printf('SIZE OF GRAD: %i x %i \n', size(grad, 1), size(grad, 2))


end
