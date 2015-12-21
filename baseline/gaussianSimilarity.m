function [ similarityScore ] = gaussianSimilarity( x1, x2, sigma )
    dif = norm(x1 - x2);
    similarityScore = exp(-(dif*dif)/(2*sigma));
end

