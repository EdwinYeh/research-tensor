function S = gaussianSimilarityMatrix(X, sigma)
%     k = 10;
    [numInstance, ~] = size(X);
    SMatrix = zeros(numInstance, numInstance);

    index = 1;
    for i = 1:numInstance
        for j = 1:numInstance
            if j >= i
                dif = norm(X(i, :) - X(j, :));
                gaussianSimilarity = exp(-(dif*dif)/(2*sigma));
                SMatrix(i, j) = gaussianSimilarity;
                SMatrix(j, i) = gaussianSimilarity;
%                 fprintf('%d, %d, %f\n', i, j, exp(-dif*dif)/(2*sigma));
                index = index + 1;
            end
        end
    end
%      plot(sortVector);
    S = SMatrix;
end