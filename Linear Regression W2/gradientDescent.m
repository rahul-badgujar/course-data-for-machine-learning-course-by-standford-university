


function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); 
J_history = zeros(num_iters, 1);
newX=[X];
for iter = 1:num_iters
    h=newX*theta;
    stdError=h-y;
    delta=(stdError'*newX)';
    theta=theta-(alpha/m)*delta;

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(newX, y, theta);

end

end
