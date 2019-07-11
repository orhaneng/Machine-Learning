function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

%1:>> X(1:3,:)


%X=   1.0000   6.1101
%	 1.0000   5.5277
%     1.0000   8.5186
%we added 1 to multiply theta0 and theta1  h(theta)= theta0*x0 + theta1*x1   
%theta =
%
%   0
%   0

% dif(1:3,:)=
%  -17.5920
%   -9.1302
%  -13.6620  
   
% we get transpose to sum the same matrix  
dif = X*theta-y
J=(dif'*dif)/(2*m)


% =========================================================================

end
