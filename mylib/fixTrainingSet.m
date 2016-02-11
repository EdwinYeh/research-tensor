function outputMatrix = fixTrainingSet(inputMatrix, labels, validateIndex)
    matrixSize = size(inputMatrix);
    numInstance = matrixSize(1);

    for i = 1:numInstance
        % If the ith instance is not the in the validation set
        % fix its label to ground truth label.
        if isempty(isempty(find(validateIndex==i)))
            inputMatrix(i, :) = 0;
            inputMatrix(i, labels(i)) = 1;
        end
    end
    outputMatrix = inputMatrix;
end