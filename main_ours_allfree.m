disp('Start training');

if isTestPhase
    resultFile = fopen(sprintf('../exp_result/result_%s.csv', exp_title), 'w');
    fprintf(resultFile, 'sigma,lambda,objectiveScore,accuracy,trainingTime,iterationUsed\n');
end

fprintf('Use Lambda:%f\n', lambda);
resultCellArray = cell(randomTryTime, 4);
bestObjectiveScore = Inf;

for t = 1: randomTryTime
    numCorrectPredict = 0;
    avgIterationUsed = 0;
    validateIndex = 1: CVFoldSize;
    foldObjectiveScores = zeros(1,numCVFold);
    TotalTimer = tic;
    totalPredictResult = zeros(numSampleInstance(targetDomain), 1);
    for fold = 1:numCVFold
        YMatrix = TrueYMatrix;
        YMatrix{targetDomain}(validateIndex, :) = zeros(CVFoldSize, numClass(targetDomain));
        W = cell(1, numDom);
        U = initU(t, :);
        V = initV(t, :);
        B = initB{t};
        iter = 0;
        diff = Inf;
        newObjectiveScore = Inf;
        
        while (diff >= 0.0001  && iter < maxIter)
            iter = iter + 1;
            oldObjectiveScore = newObjectiveScore;
            newObjectiveScore = 0;
            for i = 1:numDom
                updateTimer  = tic;
                W{i} = ones(numSampleInstance(i), numClass(i));
                if i == targetDomain
                    W{numDom}(validateIndex, :) = 0;
                end
                [projB, threeMatrixB] = SumOfMatricize(B, 2*(i - 1)+1);
                
                tmpLu = Lu{i} + diag(0.0000001*ones(numSampleInstance(i), 1));
                L = chol(tmpLu);
                
                % Solve U using closed-from
                disp('Solve U');
                U{i} = update_rule(YMatrix{i}, eye(numSampleInstance(i)), V{i}*projB', Lu{i}, W{i}, 1, U{i});
                
                %Solve V using closed-form
                disp('Solve V');
                V{i} = update_rule(YMatrix{i}, U{i}*projB, eye(numClass(i)), Lu{i}, W{i}, 0, V{i});
                
                %Update fi
                bestCPR = 20;
                CP = cp_als(tensor(threeMatrixB), bestCPR, 'printitn', 0);
                A = CP.U{1};
                E = CP.U{2};
                U3 = CP.U{3};
                
                fi = cell(1, length(CP.U{3}));
                [r, c] = size(U3);
                nextThreeB = zeros(numInstanceCluster, numFeatureCluster, r);
                sumFi = zeros(c, c);
                disp('ready');
                CPLamda = CP.lambda(:);
                for idx = 1:r
                    fi{idx} = diag(CPLamda.*U3(idx,:)');
                    sumFi = sumFi + fi{idx};
                end
                %Update A, E
                if isUpdateAE
                    %Solve A using closed-form
                    disp('Solve A');
                    A = update_rule(YMatrix{i},U{i},V{i}*E*sumFi',Lu{i},W{i},0,A);
                    
                    %Solve cvx E
                    disp('Solve E');
                    E = update_rule(YMatrix{i},U{i}*A*sumFi,V{i},Lu{i},W{i},0,E);
                    
                    for idx = 1:r
                        nextThreeB(:,:,idx) = A*fi{idx}*E';
                    end
                end
                B = InverseThreeToOriginalB(tensor(nextThreeB), 2*(i-1)+1, originalSize);
            end
            %disp(sprintf('\tCalculate this iterator error'));
            for i = 1:numDom
                [projB, ~] = SumOfMatricize(B, 2*(i - 1)+1);
                result = U{i}*projB*V{i}';
                if i == targetDomain
                    normEmp = norm((YMatrix{i} - result).*W{i}, 'fro')*norm((YMatrix{i} - result).*W{i}, 'fro');
                else
                    normEmp = norm((YMatrix{i} - result), 'fro')*norm((YMatrix{i} - result), 'fro');
                end
                smoothU = lambda*trace(U{i}'*Lu{i}*U{i});
                objectiveScore = normEmp + smoothU;
                newObjectiveScore = newObjectiveScore + objectiveScore;
            end
            %                 fprintf('iteration:%d, objectivescore:%f\n', iter, newObjectiveScore);
            diff = oldObjectiveScore - newObjectiveScore;
        end
         avgIterationUsed  = avgIterationUsed + iter/ numCVFold;
         foldObjectiveScores(fold) = newObjectiveScore;
        %calculate validationScore
        [projB, ~] = SumOfMatricize(B, 2*(targetDomain - 1)+1);
        result = U{targetDomain}*projB*V{targetDomain}';
        
        [~, maxIndex] = max(result, [], 2);
        predictResult = maxIndex;
        totalPredictResult(validateIndex) = predictResult(validateIndex);
        for i = 1: CVFoldSize
            if(predictResult(validateIndex(i)) == Label{targetDomain}(validateIndex(i)))
                numCorrectPredict = numCorrectPredict + 1;
            end
        end
        validateIndex = validateIndex + CVFoldSize;
    end
    
    accuracy = numCorrectPredict/ numSampleInstance(targetDomain);
    avgObjectiveScore = sum(foldObjectiveScores)/ numCVFold;
    avgTime = toc(TotalTimer)/ numCVFold;
    
    if avgObjectiveScore < bestObjectiveScore
        bestObjectiveScore = avgObjectiveScore;
        bestPredictResult = totalPredictResult;
    end
    
    resultCellArray{t}{1} = avgObjectiveScore;
    resultCellArray{t}{2} = accuracy*100;
    resultCellArray{t}{3} = avgTime;
    resultCellArray{t}{4} = avgIterationUsed;
    fprintf('Initial try: %d, ObjectiveScore:%f, Accuracy:%f%%\n', t, avgObjectiveScore, accuracy*100);
end

if isTestPhase
    for numResult = 1:randomTryTime
        fprintf(resultFile, '%f,%f,%f,%f,%f,%f\n', sigma, lambda, resultCellArray{numResult}{1}, resultCellArray{numResult}{2}, resultCellArray{numResult}{3}, resultCellArray{numResult}{4});
    end
    csvwrite(sprintf('../exp_result/predict_result/%s_predict_result.csv', exp_title), bestPredictResult);
    fclose(resultFile);
end

showExperimentInfo(exp_title, datasetId, prefix, numSampleInstance, numSampleFeature, numInstanceCluster, numFeatureCluster, sigma);
fprintf('done\n\n');