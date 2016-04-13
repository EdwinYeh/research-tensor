function [kernelGraph] = computeGraphInKernel(X, sigma, sigma2)
    eps = 0.00000001;
    % Compute kernal matrix of X with sigma
    kernelMatrix = gaussianSimilarityMatrix(X, sigma);
    % Add a small value to prevent divide zero problem
    kernelMatrix = kernelMatrix + eps;
    kernelDistanceMatrix = 1./ kernelMatrix;
    [maxR,maxC] = size(kernelDistanceMatrix);
    kernelGraph = zeros(maxR,maxC);
    for row = 1:maxR
        for col = 1:maxC
            distance = kernelDistanceMatrix(row, col);
            kernelGraph(row,col) = exp(-(distance*distance)/(2*sigma2));
        end
    end
end