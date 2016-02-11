function [ outputMatrix ] = minMaxNormalize( inputMatrix )
    [numRow, numCol] = size(inputMatrix);
    outputMatrix = zeros(numRow, numCol);
    for c = 1: numCol
        maxValueInColumn = max(inputMatrix(:, c));
        minValueInColumn = min(inputMatrix(:, c));
        if minValueInColumn == maxValueInColumn
            outputMatrix(:, c) = inputMatrix(:, c);
        else
            outputMatrix(:, c) = (inputMatrix(:, c) - minValueInColumn)/ (maxValueInColumn- minValueInColumn);
        end
    end
end

