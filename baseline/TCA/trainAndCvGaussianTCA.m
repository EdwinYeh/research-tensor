function [predictLabel, avgEmpError, accuracy, avgPrepareTime, avgTrainAndPredictTime ] = trainAndCvGaussianTCA( mu, sigma, numFold, numSourceData, numValidateData, numTestData, featureDimAfterReduce, sourceDomainData, targetDomainData, Y, isTestPhase)
    fprintf('mu = %f\nsigma = %f\n', mu, sigma);
    numCorrectPredict = 0;
    empErrorSum = 0;
    if isTestPhase
        sizeOfOneFold = numTestData/ numFold;
        numAllData = numSourceData + numValidateData + numTestData;
    else
        sizeOfOneFold = numValidateData/ numFold;
        numAllData = numSourceData + numValidateData;
    end
    
    % Pre-allocate matrix K, L, and compute H
    K = zeros(numAllData, numAllData);
    L = zeros(numAllData, numAllData);
    H = eye(numAllData) - ((1/(numAllData) * ones(numAllData, numAllData)));
    
    avgTrainAndPredictTime = 0;
    avgPrepareTime = 0;
    
    avgTotalTimer = tic;
    for fold = 0: (numFold-1)
        % Compute K, L matrix
        if isTestPhase
            hiddenDataIndex = (fold*sizeOfOneFold+1+numValidateData: fold*sizeOfOneFold+sizeOfOneFold+numValidateData) + numSourceData;
        else
            hiddenDataIndex = (fold*sizeOfOneFold+1: fold*sizeOfOneFold+sizeOfOneFold) + numSourceData;
        end
%         fprintf('fold: %d, holdout:(%d~%d)\n', fold, min(hiddenDataIndex), max(hiddenDataIndex));
        trainDataIndex = setdiff(1:numAllData, hiddenDataIndex);
        trainY = Y(trainDataIndex);
        validateY = Y(hiddenDataIndex);
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
                    L(i, j) = 1/ (numValidateData-sizeOfOneFold)^2;
                else
                    L(i, j) = 1/ (numSourceData*(numValidateData-sizeOfOneFold));
                end
            end
        end
        tcaMatrix = (K*L*K + mu*eye(numAllData))\(K*H*K);
        [eigVectorMatrix, ~] = eig(tcaMatrix);
        transformMatrix = eigVectorMatrix(:, 1:featureDimAfterReduce);
        % Project data in kernel space to the learned components
        dimReducedMatrix = tcaMatrix * transformMatrix;
        trainX = dimReducedMatrix(trainDataIndex, :);
        avgPrepareTime = avgPrepareTime + toc(avgTotalTimer);
        avgTrainAndPreidctTimer = tic;
        svmModel = fitcsvm(trainX, trainY, 'KernelFunction', 'gaussian');
        empErrorSum = empErrorSum + loss(svmModel, trainX, trainY);
        predictLabel = predict(svmModel, dimReducedMatrix(hiddenDataIndex, :));
        
        for i = 1: sizeOfOneFold
            if(predictLabel(i) == validateY(i))
                numCorrectPredict = numCorrectPredict + 1;
            end
        end
        avgTrainAndPredictTime = avgTrainAndPredictTime + toc(avgTrainAndPreidctTimer);
    end
    avgEmpError = empErrorSum/ numFold;
    if isTestPhase
        accuracy = numCorrectPredict/ numTestData;
    else
        accuracy = numCorrectPredict/ numValidateData;
    end
    avgTrainAndPredictTime = avgTrainAndPredictTime/ numFold;
    avgPrepareTime = avgPrepareTime/ numFold;
    disp(accuracy);
end
