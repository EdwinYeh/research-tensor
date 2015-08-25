function S = gaussianSimilarityMatrix(X, sigma)
%     k = 10;
    [numInstance, ~] = size(X);
    numNodePair = (1 + numInstance)*numInstance/2;
    SVector = zeros(1, numNodePair);
    SMatrix = zeros(numInstance, numInstance);
%     distMatrix = zeros(numInstance, numInstance);
%     sumOfKNearest = 0;
%     for i = 1:numInstance
%         for j = 1:numInstance
%             distMatrix(i, j) = norm(X(i, :) - X(j, :));
%         end
%     end
%     
%     for i = 1:numInstance
%         [sortDist, ~] = sort(distMatrix(:, i));
%         for j = 1:k
%             sumOfKNearest = sumOfKNearest + sortDist(j);
%         end
%     end
%     sigma = sumOfKNearest/ (k*numInstance);
%     fprintf('Best sigma: %f\n', sigma);
    index = 1;
    for i = 1:numInstance
        for j = 1:numInstance
            if j >= i
                dif = norm(X(i, :) - X(j, :));
                gaussianSimilarity = exp(-(dif*dif)/(2*sigma));
                SVector(index) = gaussianSimilarity;
                SMatrix(i, j) = gaussianSimilarity;
                SMatrix(j, i) = gaussianSimilarity;
                index = index + 1;
            end
        end
    end

     sortVector = sort(SVector);
     connectionThresholdIndex = numNodePair - round(numNodePair*0.03) + 1;
     connectionThreshold = sortVector(connectionThresholdIndex);
     fprintf('connection threshold = %f\n', connectionThreshold);
     plot(sortVector);
     hold on;
     plot(connectionThresholdIndex, connectionThreshold, 'X', 'color', 'r');
     
     for i = 1:numInstance
         for j = 1:numInstance
             if SMatrix(i, j) < connectionThreshold
                 SMatrix(i, j) = 0;
             end
         end
     end
    
    S = SMatrix;
end