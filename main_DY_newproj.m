disp('Start training');

if isTestPhase
    resultFile = fopen(sprintf('../exp_result/result_%s.csv', exp_title), 'a');
    fprintf(resultFile, 'sigma,lambda,objectiveScore,accuracy,trainingTime\n');
end

time = round(clock);
fprintf('Time: %d/%d/%d,%d:%d:%d\n', time(1), time(2), time(3), time(4), time(5), time(6));
fprintf('Use (Sigma, Lambda):(%f,%g)\n', sigma, lambda);
%each pair is (objective score, accuracy);
resultCellArray = cell(randomTryTime);
bestObjectiveScore = Inf;

for t = 1: randomTryTime
    numCorrectPredict = 0;
    avgIterationUsed = 0;
    validateIndex = 1: CVFoldSize;
    foldObjectiveScores = zeros(1,numCVFold);
    TotalTimer = tic;
    totalPredictResult = zeros(numSampleInstance(targetDomain), 1);
    CP1 = rand(numInstanceCluster, cpRank);
    CP2 = rand(numFeatureCluster, cpRank);
    CP3 = rand(numInstanceCluster, cpRank);
    CP4 = rand(numFeatureCluster, cpRank);
    U = initU(t, :);
    V = initV(t, :);
    for fold = 1:numCVFold
        YMatrix = TrueYMatrix;
        YMatrix{targetDomain}(validateIndex, :) = zeros(CVFoldSize, numClass(1));
        W = ones(numSampleInstance(targetDomain), numClass(1));
        W(validateIndex, :) = 0;
        iter = 0;
        diff = Inf;
        newObjectiveScore = Inf;
        
        while (diff >= 0.001  && iter < maxIter)
            iter = iter + 1;
            fprintf('Fold:%d,Iteration:%d, ObjectiveScore:%g\n', fold, iter, newObjectiveScore);
            oldObjectiveScore = newObjectiveScore;
            newObjectiveScore = 0;
            for i = 1:numDom
                [A,sumFi,E] = projectTensorToMatrix({CP1,CP2,CP3,CP4}, i);
                projB = A*sumFi*E';
                
%                 V{i} = V{i}.*sqrt((YMatrix{i}'*U{i}*projB)./((V{i}*projB'*U{i}')*U{i}*projB));
                if i == targetDomain
                    V{i} = V{i}.*sqrt((YMatrix{i}'*U{i}*projB)./((V{i}*projB'*U{i}'.*W')*U{i}*projB));
                else
                    V{i} = V{i}.*sqrt((YMatrix{i}'*U{i}*projB)./((V{i}*projB'*U{i}')*U{i}*projB));
                end
                V{i}(isnan(V{i})) = 0;
                V{i}(~isfinite(V{i})) = 0;
                
                %col normalize
                [r, ~] = size(V{i});
                for tmpI = 1:r
                    bot = sum(abs(V{i}(tmpI,:)));
                    if bot == 0
                        bot = 1;
                    end
                    V{i}(tmpI,:) = V{i}(tmpI,:)/bot;
                end
                V{i}(isnan(V{i})) = 0;
                V{i}(~isfinite(V{i})) = 0;
                tmpObjectiveScore = ShowObjective(U, V, W, YMatrix, Lu, CP1, CP2, CP3, CP4, lambda);
                if tmpObjectiveScore > oldObjectiveScore
                    fprintf('Objective increased when update V (%f=>%f)\n', oldObjectiveScore, tmpObjectiveScore);
                end
                %update U
%                 U{i} = U{i}.*sqrt((YMatrix{i}*V{i}*projB'+lambda*Su{i}*U{i})./(U{i}*projB*V{i}'*V{i}*projB'+lambda*Du{i}*U{i}));
                if i == targetDomain
                    U{i} = U{i}.*sqrt((YMatrix{i}*V{i}*projB'+lambda*Su{i}*U{i})./((U{i}*projB*V{i}'.*W)*V{i}*projB'+lambda*Du{i}*U{i}));
                else
                    U{i} = U{i}.*sqrt((YMatrix{i}*V{i}*projB'+lambda*Su{i}*U{i})./(U{i}*projB*V{i}'*V{i}*projB'+lambda*Du{i}*U{i}));
                end
                U{i}(isnan(U{i})) = 0;
                U{i}(~isfinite(U{i})) = 0;
                 
                %col normalize
                [r, ~] = size(U{i});
                for tmpI = 1:r
                    bot = sum(abs(U{i}(tmpI,:)));
                    if bot == 0
                        bot = 1;
                    end
                    U{i}(tmpI,:) = U{i}(tmpI,:)/bot;
                end
                U{i}(isnan(U{i})) = 0;
                U{i}(~isfinite(U{i})) = 0;
                 tmpObjectiveScore = ShowObjective(U, V, W, YMatrix, Lu, CP1, CP2, CP3, CP4, lambda);
                if tmpObjectiveScore > oldObjectiveScore
                    fprintf('Objective increased when update U (%f=>%f)\n', oldObjectiveScore, tmpObjectiveScore);
                end
                %update fi
                [rA, cA] = size(A);
                onesA = ones(rA, cA);
                A = A.*sqrt((U{i}'*YMatrix{i}*V{i}*E*sumFi)./(U{i}'*U{i}*A*sumFi*E'*V{i}'*V{i}*E*sumFi));
                A(isnan(A)) = 0;
                A(~isfinite(A)) = 0;
                if i == sourceDomain
                    CP1 = A;
                else
                    CP3 = A;
                end
                 tmpObjectiveScore = ShowObjective(U, V, W, YMatrix, Lu, CP1, CP2, CP3, CP4, lambda);
                if tmpObjectiveScore > oldObjectiveScore
                    fprintf('Objective increased when update A (%f=>%f)\n', oldObjectiveScore, tmpObjectiveScore);
                end
                [rE ,cE] = size(E);
                onesE = ones(rE, cE);
                E = E.*sqrt((V{i}'*YMatrix{i}'*U{i}*A*sumFi)./(V{i}'*V{i}*E*sumFi*A'*U{i}'*U{i}*A*sumFi));
                E(isnan(E)) = 0;
                E(~isfinite(E)) = 0;
                if i == sourceDomain
                   CP2 = E;
                else
                   CP4 = E;
               end
                tmpObjectiveScore = ShowObjective(U, V, W, YMatrix, Lu, CP1, CP2, CP3, CP4, lambda);
                if tmpObjectiveScore > oldObjectiveScore
                    fprintf('Objective increased when update E (%f=>%f)\n', oldObjectiveScore, tmpObjectiveScore);
                end
            end
            for i = 1:numDom
                [A,sumFi,E] = projectTensorToMatrix({CP1,CP2,CP3,CP4}, i);
                projB = A*sumFi*E';
                result = U{i}*projB*V{i}';
                if i == targetDomain
                    normEmp = norm((YMatrix{i} - result).*W, 'fro')*norm((YMatrix{i} - result).*W, 'fro');
                else
                    normEmp = norm((YMatrix{i} - result), 'fro')*norm((YMatrix{i} - result), 'fro');
                end
                smoothU = lambda*trace(U{i}'*Lu{i}*U{i});
                objectiveScore = normEmp + smoothU;
                newObjectiveScore = newObjectiveScore + objectiveScore;
            end
            diff = oldObjectiveScore - newObjectiveScore;
        end
        foldObjectiveScores(fold) = newObjectiveScore;
        
        %calculate validationScore
        [A,sumFi,E] = projectTensorToMatrix({CP1,CP2,CP3,CP4}, targetDomain);
        projB = A*sumFi*E';
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
    
    time = round(clock);
    fprintf('Time: %d/%d/%d,%d:%d:%d\n', time(1), time(2), time(3), time(4), time(5), time(6));
    fprintf('Initial try: %d, ObjectiveScore:%f, Accuracy:%f%%\n', t, avgObjectiveScore, accuracy*100);
    resultCellArray{t}{1} = avgObjectiveScore;
    resultCellArray{t}{2} = accuracy*100;
    resultCellArray{t}{3} = avgTime;
end

if isTestPhase
    for numResult = 1:randomTryTime
        fprintf(resultFile, '%f,%f,%f,%f,%f,%f\n', sigma, lambda, resultCellArray{numResult}{1}, resultCellArray{numResult}{2}, resultCellArray{numResult}{3});
    end
    %     csvwrite(sprintf('../exp_result/predict_result/%s_predict_result.csv', exp_title), bestPredictResult);
    fclose(resultFile);
end
fprintf('done\n\n');
% matlabpool close;