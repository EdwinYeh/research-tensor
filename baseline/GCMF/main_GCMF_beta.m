time = round(clock);
fprintf('Time: %d/%d/%d,%d:%d:%d\n', time(1), time(2), time(3), time(4), time(5), time(6));
fprintf('DatasetId:%d, (InstanceCluster, FeatureCluster, Sigma, Sigma2, Lambda, Gama):(%g,%g,%g,%g,%g,%g)\n', datasetId, numInstanceCluster, numFeatureCluster, sigma, sigma2, lambda, gama);

bestTestObjectiveScore = Inf;
bestTestAccuracy = 0;
bestTestTime = Inf;
U = cell(numCVFold,2);
V = cell(numCVFold,2);
H = cell(numCVFold,1);

validationAccuracyList = zeros(randomTryTime, 1);
validationObjectiveScoreList = zeros(randomTryTime, 1);
validationTimeList = zeros(randomTryTime, 1);

for t = 1: randomTryTime
    
    for fold = 1: numCVFold
        H{fold} = rand(numInstanceCluster, numFeatureCluster);
        for dom = 1: 2
            U{fold, dom} = rand(numSampleInstance(dom), numInstanceCluster);
            V{fold, dom} = rand(numSampleFeature, numFeatureCluster);
        end
    end
    
    numCorrectPredict = 0;
    hiddenIndex = 1: CVFoldSize;
    if isTestPhase
        hiddenIndex = hiddenIndex + numValidationInstance;
    end
    
    TotalTimer = tic;
    foldObjectiveScores = zeros(1,numCVFold);
    objTrack = cell(numCVFold, 1);
    
    for fold = 1:numCVFold
        revealIndex = setdiff(1:numSampleInstance(targetDomain), hiddenIndex);
        for dom = 1:numDom
            U{fold, dom} = TrueYMatrix{dom};
        end
        U{fold, targetDomain}(hiddenIndex, :) = rand(CVFoldSize, numClass(targetDomain));
        iter = 0;
        diff = Inf;
        newObjectiveScore = Inf;
        HChildCell = cell(1, numDom);
        HMotherCell = cell(1, numDom);
        stopTag = 0;
        
        while (stopTag < 50 && iter < maxIter)
            iter = iter + 1;
            %disp(diff);
            %fprintf('Fold:%d,Iteration:%d, ObjectiveScore:%g\n',fold, iter, newObjectiveScore);
            oldObjectiveScore = newObjectiveScore;
            newObjectiveScore = 0;
            for dom = 1:numDom
                V{fold,dom} = V{fold,dom}.*sqrt((X{dom}'*U{fold,dom}*H{fold}+gama*Sv{dom}*V{fold,dom})./(V{fold,dom}*H{fold}'*U{fold,dom}'*U{fold,dom}*H{fold}+gama*Dv{dom}*V{fold,dom}));
                [r, ~] = size(V{fold,dom});
                for tmpI = 1:r
                    bot = sum(abs(V{fold,dom}(tmpI,:)));
                    if bot == 0
                        bot = 1;
                    end
                    V{fold,dom}(tmpI,:) = V{fold,dom}(tmpI,:)/bot;
                end
                %Update U
                %Source domain is not updated, and target domain only data in
                %hidden index are updated
                if dom == targetDomain
                    U{fold,dom} = U{fold,dom}.*sqrt((X{dom}*V{fold,dom}*H{fold}'+lambda*Su{dom}*U{fold,dom})./(U{fold,dom}*H{fold}*V{fold,dom}'*V{fold,dom}*H{fold}'+lambda*Du{dom}*U{fold,dom}));
                    U{fold,dom}(revealIndex,:) = TrueYMatrix{dom}(revealIndex,:);
                end
                [r, ~] = size(U{fold,dom});
                for tmpI = 1:r
                    bot = sum(abs(U{fold,dom}(tmpI,:)));
                    if bot == 0
                        bot = 1;
                    end
                    U{fold,dom}(tmpI,:) = U{fold,dom}(tmpI,:)/bot;
                end
                %Update H
                HChild = zeros(numInstanceCluster, numFeatureCluster);
                HMother = zeros(numInstanceCluster, numFeatureCluster);
                for dom2 = 1:numDom
                    HChildCell{dom2} = U{fold,dom2}'*X{dom2}*V{fold,dom2};
                    HMotherCell{dom2} = U{fold,dom2}'*U{fold,dom2}*H{fold}*V{fold,dom2}'*V{fold,dom2};
                end
                for dom2 = 1:numDom
                    HChild = HChild + HChildCell{dom2};
                    HMother = HMother + HMotherCell{dom2};
                end
                H{fold} = H{fold}.*sqrt(HChild./HMother);
            end
            for dom = 1:numDom
                result = U{fold,dom}*H{fold}*V{fold,dom}';
                normEmp = norm((X{dom} - result), 'fro')*norm((X{dom} - result), 'fro');
                smoothU = lambda*trace(U{fold,dom}'*Lu{dom}*U{fold,dom});
                smoothV = gama*trace(V{fold,dom}'*Lv{dom}*V{fold,dom});
                loss = normEmp + smoothU + smoothV;
                newObjectiveScore = newObjectiveScore + loss;
                %disp(sprintf('\t\tdomain #%d => empTerm:%f, smoothU:%f, smoothV:%f ==> objective score:%f', i, normEmp, smoothU, smoothV, loss));
            end
%             fprintf('fold:%d,objective:%g\n', fold, newObjectiveScore);
            objTrack{fold} = [objTrack{fold}, newObjectiveScore];
            diff = oldObjectiveScore - newObjectiveScore;
            if diff < 1
                stopTag = stopTag + 1;
            else
                stopTag = 0;
            end
        end
        foldObjectiveScores(fold) = newObjectiveScore;
        % Make prediction
        result = U{fold,targetDomain};
        [~, maxIndex] = max(result, [], 2);
        predictResult = maxIndex;
        for cvIndex = 1: CVFoldSize
            if(predictResult(hiddenIndex(cvIndex)) == sampleLabel{targetDomain}(hiddenIndex(cvIndex)))
                numCorrectPredict = numCorrectPredict + 1;
            end
        end
        hiddenIndex = hiddenIndex + CVFoldSize;
    end
    
    if isTestPhase
        accuracy = numCorrectPredict/ numTestInstance;
    else
        accuracy = numCorrectPredict/ numValidationInstance;
    end
    
    avgObjectiveScore = sum(foldObjectiveScores)/ numCVFold;
    avgTime = toc(TotalTimer)/ numCVFold;
    
    if isTestPhase
        if accuracy > bestTestAccuracy
            bestTestObjectiveScore = avgObjectiveScore;
            bestTestAccuracy = accuracy;
            bestTestTime = avgTime;
        end
    else
        time = round(clock);
        validationAccuracyList(t) = accuracy;
        validationObjectiveScoreList(t) = avgObjectiveScore;
        validationTimeList(t) = avgTime;
    end
    %     fprintf('randomTime:%d, accuracy: %g, objectiveScore:%g\n', t, accuracy, avgObjectiveScore);
end

if isTestPhase
    fprintf(resultFile, '%g,%g,%g,%g,%g,%g,%g,%g,%g\n', numInstanceCluster, numFeatureCluster, sigma, sigma2, lambda, gama, bestTestObjectiveScore, bestTestAccuracy, bestTestTime);
else
    avgValidationAccuracy = sum(validationAccuracyList)/ randomTryTime;
    avgObjectiveScore = sum(validationObjectiveScoreList)/ randomTryTime;
    avgValidationTime = sum(validationTimeList)/ randomTryTime;
    fprintf('avgValidationAccuracy: %g, objectiveScore:%g\n', avgValidationAccuracy, avgObjectiveScore);
    compareWithTheBestGCMF(avgValidationAccuracy, avgObjectiveScore, avgValidationTime, sigma, sigma2, lambda, gama, numInstanceCluster, numFeatureCluster, resultDirectory, expTitle);
    fprintf(resultFile, '%g,%g,%g,%g,%g,%g,%g,%g,%g\n', numInstanceCluster, numFeatureCluster, sigma, sigma2, lambda, gama, avgObjectiveScore, avgValidationAccuracy, avgTime);
end
