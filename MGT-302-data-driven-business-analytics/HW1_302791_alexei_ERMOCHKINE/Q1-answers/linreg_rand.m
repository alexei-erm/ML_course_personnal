clear all; close all;

 load('weighttrain.mat')

% normalize by mean value and standard deviation

for j=1:5
    xnorm(:,j) = (x(:,j) - mean(x(:,j)) ) / std(x(:,j));
end
ynorm = (y - mean(y)) / std(y);

% constant step sizes
 alpha = 1;
 b=1000;

M = size(x,1);
theta = zeros(1,5+1); % 1 x 6
err = 0;
objfun = 0;
ITER = 0;

theta_store = theta;


for k=0:100:10000

ITER = [ITER k];
iter = k;

for i=1:iter
   
    alpha = b/(1+i); % comment this line out for constant alpha size step

    index = floor(199*rand(1)+1);
        h = theta(1) + theta(2:6) * transpose(xnorm(index,:));
        err = ( ynorm(index) - h );
    
    theta(1) = theta(1) + alpha * (1/M) * err;
    err = 0;

    for j=2:6
        index = floor(199*rand(1)+1);
        
            h = theta(1) + theta(2:6) * transpose(xnorm(index,:));
            err = ( ynorm(index) - h ) * xnorm(index,j-1);
        
        theta(j) = theta(j) + alpha * (1/M) * err;
        
    end
end

theta_store = [theta_store; theta];

for i=1:M
    h = theta(1) + theta(2:6) * transpose(xnorm(i,:));
    err = err + (h - ynorm(i))^2;
end

objfun = [objfun err/(2*M)];

end
% abs(theta-[0.0000    0.3907    0.1926   -0.1186    0.4997   -0.6876])

plot(ITER(2:end), objfun(2:end));

