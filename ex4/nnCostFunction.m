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

X= [ones(m,1), X];
A2= sigmoid(X*(Theta1'));
A2 = [ones(m,1) A2];
h = sigmoid(A2*(Theta2'));

%test=h(1:5, :)
%log_test= log(test)
%sample_y = y(1:15, :)
%-sample_y.*log(test)
%(sample_y-1).*log(1-test) 
%-sample_y.*log(test) + (sample_y-1).*log(1-test) 

y_mapped = zeros(m, num_labels);

for j=1:num_labels
  y_mapped((y==j), j) = 1;
end

%y_mapped(1:15, :)

J = sum( sum((-y_mapped.*log(h))+((y_mapped-1).*log(1-h)),2) )./m;

Theta1_first_zero = Theta1;
Theta1_first_zero(:, 1) = 0;
Theta2_frist_zero = Theta2;
Theta2_frist_zero(:, 1) = 0;

J = J + (sum(sum(Theta1_first_zero.^2)) + sum(sum(Theta2_frist_zero.^2)))*lambda/2/m;

delta_2 = zeros(size(Theta2));
delta_1 = zeros(size(Theta1));
%size_delta_1= size(delta_1)
for t=1:m
  %forwardz%%%55r%%
  a1 = X(t, :);
  z2 = a1*(Theta1');
  a2 = sigmoid(z2);
  a2 = [1 a2];
  z3 = a2*(Theta2');
  a3 = sigmoid(z3); %h
  %back
  e3 = a3 - y_mapped(t, :);
  %size(e3)
  %size(e3*Theta2)
 % size(sigmoidGradient(a2))
 
  z2=[1 z2];
  e2 = e3*Theta2.*sigmoidGradient(z2);

  %size(e2)
 % size_e3 = size(e3)
 % size_a2 = size(a2)
  delta_2 = delta_2 + (e3'*a2);
  
  e2 = e2(2:end);
 %   size_e2 = size(e2)
 %   size_a1 = size(a1)
  delta_1 = delta_1 + (e2'*a1);
end

%size(delta_2)
%size(delta_1)
%size(Theta2_grad)
%size(Theta1_grad)

first_zero_theta1 = Theta1;
first_zero_theta1(:,1)=0;
first_zero_theta2 = Theta2;
first_zero_theta2(:,1)=0;

Theta2_grad= delta_2/m + (lambda/m)*first_zero_theta2; 
Theta1_grad=delta_1/m + (lambda/m)*first_zero_theta1;










% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
