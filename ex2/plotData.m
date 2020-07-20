function plotData(X, y)
%PLOTDATA Plots the data points X and y into a new figure 
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.

% Create New Figure
figure; hold on;

% ====================== YOUR CODE HERE ======================
% Instructions: Plot the positive and negative examples on a
%               2D plot, using the option 'k+' for the positive
%               examples and 'ko' for the negative examples.
%

pos_index = (y==1);
neg_index = (y==0);

plot(X(pos_index,1), X(pos_index,2), 'g+', 'MarkerSize', 10);

plot(X(neg_index,1), X(neg_index,2), 'ro', 'MarkerSize', 10);




% =========================================================================



hold off;

end
