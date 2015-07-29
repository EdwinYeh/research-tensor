function [outputMatrix] = assignPredictResult(inputMatrix, predictResult, isMOTAR)
n = length(predictResult);
if(isMOTAR == 1)
    for i = 1:n
        if(predictResult(i) < 0.5)
            inputMatrix(i, :) = [(1 - predictResult(i)) predictResult(i) 0];
        else
            inputMatrix(i, :) = [(1 - predictResult(i)) predictResult(i) 0];
        end
    end
else
    for i = 1:n
        if(predictResult(i) < 0.5)
            inputMatrix(i, :) = [(1 - predictResult(i)) predictResult(i)];
        else
            inputMatrix(i, :) = [(1 - predictResult(i)) predictResult(i)];
        end
    end
end
outputMatrix = inputMatrix;
end