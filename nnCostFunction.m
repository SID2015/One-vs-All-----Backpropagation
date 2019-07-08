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
         
% We need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== OUR CODE HERE ======================
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, we can verify that our
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. We should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, we can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. We need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: We can implement this around the code for
%               backpropagation. That is, we can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


eye_matrix = eye(num_labels);
y_matrix = eye_matrix(y,:);

x_column = ones(m, 1);

X = [x_column ,X];
h1 = (X)*(Theta1');

h_sigm1 = sigmoid(h1);

h_sigm1_column  = ones(m, 1);

h_sigm1 = [h_sigm1_column , h_sigm1];

h2 = (h_sigm1)*(Theta2');

h_sigm2 = sigmoid(h2);

left_summation = ((y_matrix)')*(log(h_sigm2));

right_summation = ((1-y_matrix)')*(log(1-h_sigm2));

summation_cost = left_summation + right_summation ;

summation_cost = -((1/m)*(summation_cost));

J = trace(summation_cost);


%Regularization_Term

Theta1_New = Theta1;
Theta2_New = Theta2;

Theta1_New(:,1)=[];
Theta2_New(:,1)=[];
summ_reg_1 = (Theta1_New)*(Theta1_New');  %((y_matrix)')*(log(h_sigm2));

summ_reg_2 =  (Theta2_New')*(Theta2_New);  %((1-y_matrix)')*(log(1-h_sigm2));

Totalsum_with_reg = summ_reg_1 + summ_reg_2;

Trace_sum = trace(Totalsum_with_reg);

Trace_reg = (lambda/(2*m))*(Trace_sum);

J = J + Trace_reg;


%%Backpropagation

%X(:,402) =[];
%X(:,401) =[];


d3  = h_sigm2 - y_matrix;

z2 = h1;

u = (sigmoid(z2)).*(1.0 - (sigmoid(z2)));

sigm_grad = u;

%New_Theta2 = Theta2(:,2:end);

d2 = d3*Theta2_New;

d2 = (d2).*sigm_grad;

Delta1 = (d2')*X;

Delta2 = (d3')*h_sigm1;

Theta1_grad = (1/m)*(Delta1);

Theta2_grad = (1/m)*(Delta2);

Theta1(:,1) = 0;

Theta2(:,1) = 0;

Theta1_grad  = Theta1_grad + (lambda/m)*(Theta1);

Theta2_grad = Theta2_grad + (lambda/m)*(Theta2);

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
