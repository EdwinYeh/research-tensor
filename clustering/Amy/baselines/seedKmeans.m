% X is a N*F matrix, where N = number of node, F = number of feature
% cNum is the number of cluster
% seeds is a N*1 matrix, where N = number of nodes, indicate the cluster
% label of seeds. 0 means the node doesn't have the
% constrain
function y = seedKmeans(X,cNum, seeds)

N = length(X(:,1));
F = length(X(1,:));
y = zeros(N,1);
centroid = zeros(cNum, F);
seedNumForEachCentroid = zeros(cNum,1);

% init centroid based on seeds
for i=1:N
    if(seeds(i) ~= 0)
        centroid(seeds(i),:) = centroid(seeds(i),:) + X(i,:);
        seedNumForEachCentroid(seeds(i),1) = seedNumForEachCentroid(seeds(i),1) + 1;
    end
end
centroid = bsxfun(@rdivide,centroid,seedNumForEachCentroid);

% start iteration
delta = Inf;
stopThreshold = 0.1;
while(delta > stopThreshold)
    % assign cluster
    for i=1:N
        minIndex = 1;
        min = Inf;
        if(seeds(i) ~= 0)
            y(i) = seeds(i); % keep seeds' clusters
            continue;
        end
        for j=1:cNum
            dist = distance(centroid(j,:), X(i,:));
            if(dist < min)
                min = dist;
                minIndex = j;
            end
        end
        y(i) = minIndex;
    end
    
    % recalculate centroid
    newCentroid = zeros(cNum,F);
    memberCount = zeros(cNum,1);
    
    for i=1:N
        newCentroid(y(i),:) = newCentroid(y(i),:) + X(i,:);
        memberCount(y(i)) = memberCount(y(i)) + 1;
    end
    newCentroid = bsxfun(@rdivide,newCentroid,memberCount);
    
    % stop criteria
    delta = 0;
    for i=1:cNum
        d = distance(centroid(i,:), newCentroid(i,:));
        if(d > delta)
            delta = d;
        end
    end
    centroid = newCentroid;
    
end

end


function d = distance(input1,input2)
d = norm(input1-input2);
end