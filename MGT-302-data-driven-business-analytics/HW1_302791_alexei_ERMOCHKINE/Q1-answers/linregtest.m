close all; clear all;

load('weighttest.mat');

% theta obtained in linreg.m 
theta = [0.0000    0.3907    0.1926   -0.1186    0.4997   -0.6876];

err = 0;

for j=1:5
    xnorm(:,j) = (x(:,j) - mean(x(:,j)) ) / std(x(:,j));
end
ynorm = (y - mean(y)) / std(y);

yp = theta(1) + theta(2:6) * transpose(xnorm);
yp = transpose(yp);

% max(abs(yp-ynorm))

for i=1:200
    h = theta(1) + theta(2:6) * transpose(xnorm(i,:));
    err = err + (h - ynorm(i))^2;
end

objfuntest = err/(2*200)