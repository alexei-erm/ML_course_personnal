close all; clear all;
load('data1.mat');


plotData(X,y);

N = 5;

for iter=0:N

model = svmTrain(X, y, 1, @linearKernel, 1e-3, 20);

hold on; visualizeBoundaryLinear(X, y, model);


end

