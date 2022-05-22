function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%
% X : 5000 x 400
% Theta1: 25 x 401
% Theta2: 10 x 26
input = [ones(m, 1), X]; %5000 x 401

layer_2 = input * Theta1'; %5000 x 25
layer_2 = sigmoid(layer_2);
layer_2 = [ones(m, 1), layer_2]; %5000 x 26

layer_3 = layer_2 * Theta2'; %5000 x 10
layer_3 = sigmoid(layer_3);

[num, p] = max(layer_3, [], 2);


% =========================================================================


end
