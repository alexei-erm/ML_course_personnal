close all; clear all;

load('kmeans.mat'); x = kmeans; clear kmeans;

        K = 2;         %  desired # of clusters

M = size(x,1); % # of data points
N = size(x,2); % # of features 

% normalize by mean value and standard deviation
for j=1:N
    x(:,j) = (x(:,j) - mean(x(:,j)) ) / std(x(:,j));
end

% mu = zeros(K,N); % will contain the K centroids at any given iteration
S = zeros(M,K);  % j-th column will contain indices of datapoints belonging to j-th cluster
c = zeros(1,M);  % contains all distance between x(i) and the closest centroid, then labels for data


% Initialising: 
% we chose K random points out of the dataset as initial centroids, 
% to which we add a small random displacement to make sure
% every initial centroid is distinct
mu = x(floor(M*rand(1,K)+1),:) + 0.01*rand(K,N);

iter = 100; % desired # of iterations

for n=1:iter
    S = zeros(M,K);
    S_index = ones(1,K); % counter that keeps track of filled slots in S

    for i=1:M

        % finding closest centroid and its distance to x(i)
        
        cc = zeros(1,K);
        for j=1:K
            cc(j) = norm(x(i,:)-mu(j,:));
        end
        c(i) = min(cc);

        % assigning x(i) to cluster j

        jj = find(cc==min(cc));
        S(S_index(jj), jj) = i;
        S_index(jj) = S_index(jj)+1;
    end

    % updating centroids mu(j)

    for j=1:K
        division=0;
        for i=1:M
            logic = ~isempty(find(S(:,j)==i));
            mu(j,:) = mu(j,:) + logic * x(i,:);
            division = division + logic;
        end
        if(division==0)   % avoiding division by 0 
            division = 1;
        end
        mu(j,:) = mu(j,:) / division;
    end

end

% plotting data with different colors representing different clusters 

scatter(mu(:,1),mu(:,2),'ko','LineWidth',2);
CM = jet(K);
for j=1:K
    Sj = transpose(S(find(S(:,j)~=0),j));
    hold on;
    scatter(x(Sj,1),x(Sj,2),'color',CM(j,:),'marker','.');
end

objfun = sum(c)




%% this section is for plotting different values of the cost function with increasing K

close all; clear all;

load('kmeans.mat'); x = kmeans; clear kmeans;

objfun = 0; Kstore = 0;
for K =[1 2 4 8 11:10:51]       %  desired # of clusters
    Kstore = [Kstore K];

M = size(x,1); % # of data points
N = size(x,2); % # of features 

% normalize by mean value and standard deviation
for j=1:N
    x(:,j) = (x(:,j) - mean(x(:,j)) ) / std(x(:,j));
end

% mu = zeros(K,N); % will contain the K centroids at any given iteration
S = zeros(M,K);  % j-th column will contain indices of datapoints belonging to j-th cluster
c = zeros(1,M);  % contains all distance between x(i) and the closest centroid, then labels for data


% Initialising: 
% we chose K random points out of the dataset as initial centroids, 
% to which we add a small random displacement to make sure
% every initial centroid is distinct
mu = x(floor(M*rand(1,K)+1),:) + 0.01*rand(K,N);

iter = 100; % desired # of iterations

for n=1:iter
    S = zeros(M,K);
    S_index = ones(1,K); % counter that keeps track of filled slots in S

    for i=1:M

        % finding closest centroid and its distance to x(i)
        
        cc = zeros(1,K);
        for j=1:K
            cc(j) = norm(x(i,:)-mu(j,:));
        end
        c(i) = min(cc);

        % assigning x(i) to cluster j

        jj = find(cc==min(cc));
        S(S_index(jj), jj) = i;
        S_index(jj) = S_index(jj)+1;
    end

    % updating centroids mu(j)

    for j=1:K
        division=0;
        for i=1:M
            logic = ~isempty(find(S(:,j)==i));
            mu(j,:) = mu(j,:) + logic * x(i,:);
            division = division + logic;
        end
        if(division==0)   % avoiding division by 0 
            division = 1;
        end
        mu(j,:) = mu(j,:) / division;
    end

end

objfun = [objfun sum(c)];
end

plot(Kstore(2:end),objfun(2:end),'rx-'); 

% empirical relation between the cost function and K 
% is of the form of an offset inverse f(x)=A+(B/x)

hold on; X = linspace(1,50,100); plot(X, 60+(290./X),'b');

