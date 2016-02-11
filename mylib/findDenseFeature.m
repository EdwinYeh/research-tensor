function [ vectorOfDenseFeature ] = findDenseFeature(inputMatrix, numSampleFeature )
    [~, numFeature] = size(inputMatrix);
    sumM = sum(inputMatrix);
    [~, newIndex] = sort(sumM);
    vectorOfDenseFeature = newIndex(numFeature-numSampleFeature+1:numFeature);
end