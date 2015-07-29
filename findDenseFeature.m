function [ vectorOfDenseFeature ] = findDenseFeature( sourceMatrix, targetMatrix, numSampleFeature )
    %disp('finding dense feature.');
    [~, numSourceFeature] = size(sourceMatrix);
    [~, numTargetFeature] = size(targetMatrix);
    if numSourceFeature > numTargetFeature
        sourceMatrix = sourceMatrix(:, 1:numTargetMatrix);
        numFeature = numTargetFeature;
    elseif numTargetFeature > numSourceFeature
        targetMatrix = targetMatrix(:, 1:numSourceFeature);
        numFeature = numSourceFeature;
    end
    M = [sourceMatrix; targetMatrix];
    sumM = sum(M);
    [~, oriIndex] = sort(sumM);
    %disp('dense feature found.');
    vectorOfDenseFeature = oriIndex(numFeature-numSampleFeature+1:numFeature);
end