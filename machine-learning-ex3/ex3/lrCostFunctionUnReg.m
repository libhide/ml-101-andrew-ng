function [J, grad] = lrCostFunctionUnReg(theta, X, y)
    m = length(y); % number of training examples
    
    component1 = -1 * (y .* log(sigmoid(X * theta)));
    component2 = (1 - y) .* log(1 - sigmoid(X * theta));
 
    J = sum(component1 - component2) / m;
 
    grad = X' * (sigmoid(X * theta) - y) / m;
end

