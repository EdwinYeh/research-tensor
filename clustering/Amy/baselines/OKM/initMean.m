function means = initMean (X, K, seeds)

[Ndata, dims] = size(X);
means = zeros(K, dims);

if (nargin ==2)
    for i=1:K-1
       means(i,:) = X(i,:);
    end
    means(K,:) = mean(X(i+1:Ndata,:));
    
elseif (nargin==3)
    [Kseed,~] = size(seeds);
    Nseed = sum(sum(seeds));
    for i=1:Kseed
       if sum(seeds(i,:)) == 0
           randp = randperm(Ndata);
           tempSeeds = zeros(K,Ndata);
           tempSeeds(i,randp(1:2)) = 1;
           means(i,:) = mean(X(logical(tempSeeds(i,:)),:));
       else
           means(i,:) = mean(X(logical(seeds(i,:)),:));
       end
    end
    nonSeedX = X(~logical(sum(seeds)),:);
    if K>Kseed
        display('warning: K > Kseed');
        for i=Kseed+1:K-1
           means(i,:) = nonSeedX((i-Kseed),:);
        end
        means(K,:) = mean(nonSeedX(K-Kseed:Ndata-Nseed,:));
    end
else
   disp('error in initMean: input parameter') 
end
end