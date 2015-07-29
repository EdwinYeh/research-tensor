function outputMatrix = fixTrainingSet(inputMatrix, labels, validateIndex)
    %UNTITLED Summary of this function goes here
    %   Detailed explanation goes here
    matrixSize = size(inputMatrix);
    userSize = matrixSize(1);
    %userSize
    for i = 1:userSize
        if isempty(isempty(find(validateIndex==i)))
            %fprintf('non-test data:%d\n', i);
            inputMatrix(i, :) = 0;
            inputMatrix(i, labels(i)) = 1;
        end
    end
    outputMatrix = inputMatrix;
end

