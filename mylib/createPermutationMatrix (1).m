function [permutationMatrix] = createPermutationMatrix(numRow,numCol)
    maxLength = max(numRow, numCol);
    V = (1:maxLength)';
    Vstar = V(randperm(length(V)));
    P = bsxfun(@eq, V', Vstar);
    permutationMatrix = P(1:numRow, 1:numCol);
    permutationMatrix = double(permutationMatrix);
end