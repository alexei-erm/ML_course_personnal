close all; clear all;

load('weighttrain.mat')

% normalize by mean value and standard deviation
for j=1:5
    xnorm(:,j) = (x(:,j) - mean(x(:,j)) ) / std(x(:,j));
end
ynorm = (y - mean(y)) / std(y);

alpha = 1; % fastest convergence for alpha = 1 (in 2-3 iterations!)

M = size(x,1); % size of sample
theta = zeros(1,5+1); % 1 x 6
err = 0; 
objfun = 0;
ITER = 0;

theta_store = theta;


for k=1:20

ITER = [ITER k];
iter = k;

for i=1:iter
    
    for i=1:M
        h = theta(1) + theta(2:6) * transpose(xnorm(i,:));
        err = err + ( ynorm(i) - h );
    end
    theta(1) = theta(1) + alpha * (1/M) * err;
    err = 0;

    for j=2:6
        for i=1:M
            h = theta(1) + theta(2:6) * transpose(xnorm(i,:));
            err = err + ( ynorm(i) - h ) * xnorm(i,j-1);
        end
        theta(j) = theta(j) + alpha * (1/M) * err;
        err = 0;
    end
end

theta_store = [theta_store; theta];

for i=1:M
    h = theta(1) + theta(2:6) * transpose(xnorm(i,:));
    err = err + (h - ynorm(i))^2;
end

objfun = [objfun err/(2*M)];

end

plot(ITER(2:end), objfun(2:end));
