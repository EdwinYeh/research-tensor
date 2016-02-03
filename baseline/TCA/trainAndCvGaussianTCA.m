function [predictLabel, avgEmpError, accuracy ] = trainAndCvGaussianTCA( mu, sigma, numFold, numSourceData, numTargetData, featureDimAfterReduce, sourceDomainData, targetDomainData, Y)
    fprintf('mu = %f\nsigma = %f\n', mu, sigma);
    numCorrectPredict = 0;
    empErrorSum = 0;
    sizeOfOneFold = numTargetData/ numFold;
    numAllData = numSourceData + numTargetData;
    
    % Pre-allocate matrix K, L, and compute H
    K = zeros(numAllData, numAllData);
    L = zeros(numAllData, numAllData);
    H = eye(numAllData) - ((1/(numAllData) * ones(numAllData, numAllData)));
    
    for fold = 0: (numFold-1)
        % fprintf('fold: %d\n', fold);
        % Compute K, L matrix
        validateDataIndex = (fold*sizeOfOneFold+1: fold*sizeOfOneFold+sizeOfOneFold) + numSourceData;
        trainDataIndex = setdiff(1:numAllData, validateDataIndex);
        trainY = Y(trainDataIndex);
        validateY = Y(validateDataIndex);
        %             fprintf('testIndex: %d~%d\n', min(testDataIndex), max(testDataIndex));
        
        for i = 1:numAllData
            for j = 1:numAllData
                if i > numSourceData
                    instance1 = targetDomainData(i-numSourceData, :);
                else
                    instance1 = sourceDomainData(i, :);
                end
                
                if j > numSourceData
                    instance2 = targetDomainData(j-numSourceData, :);
                else
                    instance2 = sourceDomainData(j, :);
                end
                
                %linear kernal
                %K(i, j) = instance1 * instance2';
                K(i, j) = gaussianSimilarity(instance1, instance2, sigma);
                
                if i < numSourceData && j < numSourceData
                    L(i, j) = 1/ numSourceData^2;
                elseif i > numSourceData && j > numSourceData
                    L(i, j) = 1/ (numTargetData-sizeOfOneFold)^2;
                else
                    L(i, j) = 1/ (numSourceData*(numTargetData-sizeOfOneFold));
                end
            end
        end
        tcaMatrix = (K*L*K + mu*eye(numAllData))\(K*H*K);
        [eigVectorMatrix, ~] = eig(tcaMatrix);
        transformMatrix = eigVectorMatrix(:, 1:featureDimAfterReduce);
        % Project data in kernel space to the learned components
        dimReducedMatrix = tcaMatrix * transformMatrix;
        trainX = dimReducedMatrix(trainDataIndex, :);
        svmModel = fitcsvm(trainX, trainY, 'KernelFunction', 'linear');
        empErrorSum = empErrorSum + loss(svmModel, trainX, trainY);
        predictLabel = predict(svmModel, dimReducedMatrix(validateDataIndex, :));
        
        for i = 1: sizeOfOneFold
            if(predictLabel(i) == validateY(i))
                numCorrectPredict = numCorrectPredict + 1;
            end
        end
    end
    avgEmpError = empErrorSum/ numFold;
    accuracy = numCorrectPredict/ numTargetData;
    disp(avgEmpError);
    disp(accuracy);
end