disp('Start training');

resultFile = fopen(sprintf('../exp_result/%s.csv', exp_title), 'a');
fprintf(resultFile, 'sigma,sigma2,lambda,delta,objectiveScore,accuracy,trainingTime\n');

time = round(clock);
fprintf('Time: %d/%d/%d,%d:%d:%d\n', time(1), time(2), time(3), time(4), time(5), time(6));
fprintf('Use (Sigma, Sigma2, Lambda, Delta):(%g,%g,%g,%g)\n', sigma, sigma2, lambda, delta);
%each pair is (objective score, accuracy);
resultCellArray = cell(randomTryTime);
bestObjectiveScore = Inf;
bestAccuracy = 0;
bestTime = 0;
SU=cell(1,2);
SV=cell(1,2);
U = cell(2,1);
V = cell(2,1);
realU = cell(2,1);
realV = cell(2,1);

for t = 1: randomTryTime

    CP1 = rand(numInstanceCluster, cpRank);
    CP2 = rand(numFeatureCluster, cpRank);
    CP3 = rand(numInstanceCluster, cpRank);
    CP4 = rand(numFeatureCluster, cpRank);

    for i = 1:2
        U{i} = rand(numSampleInstance(i), numInstanceCluster);
        V{i} = rand(2, numFeatureCluster);
    end

    realCP1 = rand(numInstanceCluster, cpRank);
    realCP2 = rand(numFeatureCluster, cpRank);
    realCP3 = rand(numInstanceCluster, cpRank);
    realCP4 = rand(numFeatureCluster, cpRank);

    for i = 1:2
        realU{i} = rand(numSampleInstance(i), numInstanceCluster);
        realV{i} = rand(2, numFeatureCluster);
    end

    for fakeOptimization = 1:2
        numCorrectPredict = 0;
        validateIndex = 1: CVFoldSize;
        TotalTimer = tic;
        foldObjectiveScores = zeros(1,numCVFold);
        if fakeOptimization == 2
            CP1 = realCP1;
            CP2 = realCP2;
            CP3 = realCP3;
            CP4 = realCP4;
            U = realU;
            V = realV;
        end

        for fold = 1:numCVFold
            YMatrix = TrueYMatrix;
            YMatrix{targetDomain}(validateIndex, :) = zeros(CVFoldSize, numClass(1));
            W = ones(numSampleInstance(targetDomain), numClass(1));
            W(validateIndex, :) = 0;
            [rY,cY]=size(YMatrix{1});
            SU{1}=eye(rY);
            SU{2}=SU{1};
            SV{1}=eye(cY);
            SV{2}=SV{1};
            iter = 0;
            diff = Inf;
            newObjectiveScore = Inf;

            while (diff >= 0.001  && iter < maxIter)
                iter = iter + 1;
                fprintf('Fold:%d,Iteration:%d, ObjectiveScore:%g\n', fold, iter, newObjectiveScore);
                oldObjectiveScore = newObjectiveScore;
                tmpOldObj=oldObjectiveScore;
                newObjectiveScore = 0;
                for i = 1:numDom
                    [A,sumFi,E] = projectTensorToMatrix({CP1,CP2,CP3,CP4}, i);
                    projB = A*sumFi*E';

                    if i == targetDomain
                        V{i} = V{i}.*sqrt(((YMatrix{i}.*W)'*U{i}*projB)./(V{i}*V{i}'*(V{i}*projB'*U{i}'.*W')*U{i}*projB));
                    else
                        V{i} = V{i}.*sqrt((YMatrix{i}'*U{i}*projB)./(V{i}*V{i}'*(V{i}*projB'*U{i}')*U{i}*projB));
                    end
                    V{i}(isnan(V{i})) = 0;
                    V{i}(~isfinite(V{i})) = 0;

                    tmpObjectiveScore = ShowObjectiveS(SU,SV,U, V, W, YMatrix, Lu, CP1, CP2, CP3, CP4, lambda);
                    if tmpObjectiveScore > tmpOldObj
                        fprintf('Domain:%d, Objective increased when update V (%f=>%f)\n', i, tmpOldObj, tmpObjectiveScore);
                    end
                    tmpOldObj=tmpObjectiveScore;
                    %update U
                    if i == targetDomain
                        U{i} = U{i}.*sqrt(((YMatrix{i}.*W)*V{i}*projB'+lambda*Su{i}*U{i})./(U{i}*U{i}'*(U{i}*projB*V{i}'.*W)*V{i}*projB'+lambda*Du{i}*U{i}));
                    else
                        U{i} = U{i}.*sqrt((YMatrix{i}*V{i}*projB'+lambda*Su{i}*U{i})./(U{i}*U{i}'*U{i}*projB*V{i}'*V{i}*projB'+lambda*Du{i}*U{i}));
                    end
                    U{i}(isnan(U{i})) = 0;
                    U{i}(~isfinite(U{i})) = 0;

                    tmpObjectiveScore = ShowObjectiveS(SU,SV,U, V, W, YMatrix, Lu, CP1, CP2, CP3, CP4, lambda);
                    if tmpObjectiveScore > tmpOldObj
                        fprintf('Domain:%d, Objective increased when update U (%f=>%f)\n', i, tmpOldObj, tmpObjectiveScore);
                    end
                    tmpOldObj=tmpObjectiveScore;
                    %update AE
                    [rA, cA] = size(A);
		    [rE, cE] = size(E);
                    onesA = ones(rA, cA);
                    if i ==targetDomain
                        A = A.*sqrt((U{i}'*(YMatrix{i}.*W)*V{i}*E*sumFi)./(U{i}'*(U{i}*A*sumFi*E'*V{i}'.*W)*V{i}*E*sumFi+delta*ones(rE,rA)'*(sumFi*E')'));
                    else
                        A = A.*sqrt((U{i}'*YMatrix{i}*V{i}*E*sumFi)./(U{i}'*U{i}*A*sumFi*E'*V{i}'*V{i}*E*sumFi+delta*ones(rE,rA)'*(sumFi*E')'));
                    end

                    A(isnan(A)) = 0;
                    A(~isfinite(A)) = 0;
                    if i == sourceDomain
                        CP1 = A;
                    else
                        CP3 = A;
                    end
                    tmpObjectiveScore = ShowObjectiveS(SU,SV,U, V, W, YMatrix, Lu, CP1, CP2, CP3, CP4, lambda);
                    if tmpObjectiveScore > tmpOldObj
                        fprintf('Domain:%d, Objective increased when update A (%f=>%f)\n', i, tmpOldObj, tmpObjectiveScore);
                    end
                    tmpOldObj = tmpObjectiveScore;
                    [rE ,cE] = size(E);
                    onesE = ones(rE, cE);
                    if i == targetDomain
                        E = E.*sqrt((V{i}'*(YMatrix{i}.*W)'*U{i}*A*sumFi)./(V{i}'*(V{i}*E*sumFi*A'*U{i}'.*W')*U{i}*A*sumFi+delta*ones(rE,rA)*A*sumFi));
                    else
                        E = E.*sqrt((V{i}'*YMatrix{i}'*U{i}*A*sumFi)./(V{i}'*V{i}*E*sumFi*A'*U{i}'*U{i}*A*sumFi+delta*ones(rE,rA)*A*sumFi));
                    end

                    E(isnan(E)) = 0;
                    E(~isfinite(E)) = 0;
                    if i == sourceDomain
                        CP2 = E;
                    else
                        CP4 = E;
                    end
                    tmpObjectiveScore = ShowObjectiveS(SU,SV,U, V, W, YMatrix, Lu, CP1, CP2, CP3, CP4, lambda);
                    if tmpObjectiveScore > tmpOldObj
                        fprintf('Domain:%d, Objective increased when update E (%f=>%f)\n', i, tmpOldObj, tmpObjectiveScore);
                    end
                    tmpOldObj=tmpObjectiveScore;
                end
                newObjectiveScore = ShowObjectiveS(SU,SV,U, V, W, YMatrix, Lu, CP1, CP2, CP3, CP4, lambda);
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

        if fakeOptimization == 1
            realCP1 = CP1;
            realCP2 = CP2;
            realCP3 = CP3;
            realCP4 = CP4;
            realU = U;
            realV = V;
        end

        if fakeOptimization == 2
            accuracy = numCorrectPredict/ numSampleInstance(targetDomain);
            avgObjectiveScore = sum(foldObjectiveScores)/ numCVFold;
            avgTime = toc(TotalTimer)/ numCVFold;

            if avgObjectiveScore < bestObjectiveScore
                bestObjectiveScore = avgObjectiveScore;
                bestAccuracy = accuracy*100;
                bestTime = avgTime;
                bestPredictResult = totalPredictResult;
            end
            time = round(clock);
            fprintf('Time: %d/%d/%d,%d:%d:%d\n', time(1), time(2), time(3), time(4), time(5), time(6));
            fprintf('Initial try: %d, ObjectiveScore:%f, Accuracy:%f%%\n', t, avgObjectiveScore, accuracy*100);
        end
    end
end

fprintf(resultFile, '%g,%g,%g,%g,%g,%g,%g\n', sigma, sigma2, lambda, delta, bestObjectiveScore, bestAccuracy, bestTime);
%     csvwrite(sprintf('../exp_result/predict_result/%s_predict_result.csv', exp_title), bestPredictResult);
fclose(resultFile);
fprintf('done\n\n');
% matlabpool close;
