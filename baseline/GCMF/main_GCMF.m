disp('Start training');

if isTestPhase
    resultFile = fopen(sprintf('result_%s.csv', exp_title), 'a');
    fprintf(resultFile, 'sigma,gama,lambda,objectiveScore,accuracy,trainingTime\n');
end

fprintf('Use Lambda: %f, Gama: %f\n', lambda, gama);
resultCellArray = cell(randomTryTime, 3);
bestObjectiveScore = Inf;

for t = 1: randomTryTime
    numCorrectPredict = 0;
    avgIterationUsed = 0;
    validateIndex = 1: CVFoldSize;
    foldObjectiveScores = zeros(1,numCVFold);
    TotalTimer = tic;
    totalPredictResult = zeros(numSampleInstance(targetDomain), 1);
    for fold = 1:numCVFold
        %re-initialize
        U = initU(t, :);
        V = initV(t, :);
        H = initH{t};
        for i = 1:numDom
            if i == targetDomain
                U{i} = TrueYMatrix{i};
                U{i}(validateIndex, :) = zeros(CVFoldSize, numClass(i));
            else
                U{i} = TrueYMatrix{i};
            end
        end
        HChildCell = cell(1, numDom);
        HMotherCell = cell(1, numDom);
        %Iterative update
        newObjectiveScore = Inf;
        oldObjectiveScore = Inf;
        iter = 0;
        diff = Inf;
        
        while (diff >= abs(0.001)  && iter < maxIter)%(abs(ObjectiveScore - newObjectiveScore) >= 0.1 && iter < maxIter)
            iter = iter + 1;
            oldObjectiveScore = newObjectiveScore;
            %disp(sprintf('\t#Iterator:%d', iter));
            %disp(newObjectiveScore);
            newObjectiveScore = 0;
            for i = 1:numDom
                %disp(sprintf('\t\tupdate V...'));
                %update V
                V{i} = V{i}.*sqrt((X{i}'*U{i}*H+gama*Sv{i}*V{i})./(V{i}*H'*U{i}'*U{i}*H+gama*Dv{i}*V{i}));
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
                
                %disp(sprintf('\t\tupdate U...'));
                %update U
                if(i == targetDomain)
                    U{i} = U{i}.*sqrt((X{i}*V{i}*H'+lambda*Su{i}*U{i})./(U{i}*H*V{i}'*V{i}*H'+lambda*Du{i}*U{i}));
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
                end
                
                %update H
                HChild = zeros(numInstanceCluster, numFeatureCluster);
                HMother = zeros(numInstanceCluster, numFeatureCluster);
                for j = 1:numDom
                    HChildCell{j} = U{j}'*X{j}*V{j};
                    HMotherCell{j} = U{j}'*U{j}*H*V{j}'*V{j};
                end
                for j = 1:numDom
                    HChild = HChild + HChildCell{j};
                    HMother = HMother + HMotherCell{j};
                end
                H = H.*sqrt(HChild./HMother);
            end
            avgIterationUsed  = avgIterationUsed + iter/ numCVFold;
            foldObjectiveScores(fold) = newObjectiveScore;
            %disp(sprintf('\tCalculate this iterator error'));
            for i = 1:numDom
                result = U{i}*H*V{i}';
                normEmp = norm((X{i} - result), 'fro')*norm((X{i} - result), 'fro');
                smoothU = lambda*trace(U{i}'*Lu{i}*U{i});
                smoothV = gama*trace(V{i}'*Lv{i}*V{i});
                loss = normEmp + smoothU + smoothV;
                newObjectiveScore = newObjectiveScore + loss;
                %disp(sprintf('\t\tdomain #%d => empTerm:%f, smoothU:%f, smoothV:%f ==> objective score:%f', i, normEmp, smoothU, smoothV, loss));
            end
            %disp(sprintf('\tEmperical Error:%f', newObjectiveScore));
            %fprintf('iter:%d, error = %f\n', iter, newObjectiveScore);
            diff = oldObjectiveScore - newObjectiveScore;
        end
        foldObjectiveScores(fold) = newObjectiveScore;
        %calculate validationScore
        [~, maxIndex] = max(U{targetDomain}, [], 2);
        predictResult = maxIndex;
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
    
    fprintf('Initial try: %d, ObjectiveScore:%f, Accuracy:%f%%\n', t, avgObjectiveScore, accuracy*100);
end

if isTestPhase
    for numResult = 1:randomTryTime
        fprintf(resultFile, '%f,%f,%f,%f,%f,%f\n', sigma, gama, lambda, resultCellArray{numResult}{1}, resultCellArray{numResult}{2}, resultCellArray{numResult}{3});
    end
    fclose(resultFile);
end

showExperimentInfo(exp_title, datasetId, prefix, numSampleInstance, numSampleFeature, numInstanceCluster, numFeatureCluster, sigma);
fprintf('done\n');