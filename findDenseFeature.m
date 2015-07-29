function [ vectorOfDenseFeature ] = findDenseFeature(inputMatrix, numSampleFeature )
    [~, numFeature] = size(inputMatrix);
    sumM = sum(inputMatrix);
    [~, oriIndex] = sort(sumM);
    %disp('dense feature found.');
    vectorOfDenseFeature = oriIndex(numFeature-numSampleFeature+1:numFeature);
end