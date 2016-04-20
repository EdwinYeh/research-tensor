time = round(clock);
fprintf('Time: %d/%d/%d,%d:%d:%d\n', time(1), time(2), time(3), time(4), time(5), time(6));
fprintf('Use (Sigma, Sigma2, Lambda, Delta):(%g,%g,%g,%g)\n', sigma, sigma2, lambda, delta);

bestObjectiveScore = Inf;
bestAccuracy = 0;
bestTime = 0;
SU=cell(1,2);
SV=cell(1,2);
U = cell(1,2);
V = cell(1,2);
tmpU = cell(1,2);
tmpV = cell(1,2);
realU = cell(2,1);
realV = cell(2,1);

for t = 1: randomTryTime

    CP1 = rand(numInstanceCluster, cpRank);
    CP2 = rand(numFeatureCluster, cpRank);
    CP3 = rand(numInstanceCluster, cpRank);
    CP4 = rand(numFeatureCluster, cpRank);
    
    for dom = 1:2
        U{dom} = rand(numSampleInstance(dom), numInstanceCluster);
        V{dom} = rand(2, numFeatureCluster);
    end

    realCP1 = rand(numInstanceCluster, cpRank);
    realCP2 = rand(numFeatureCluster, cpRank);
    realCP3 = rand(numInstanceCluster, cpRank);
    realCP4 = rand(numFeatureCluster, cpRank);

    for dom = 1:2
        realU{dom} = rand(numSampleInstance(dom), numInstanceCluster);
        realV{dom} = rand(2, numFeatureCluster);
    end
    
    % When fakeOptimization == 1, train UVAE  to be the initial points
    % during fakeOptimization == 2. Only the report the result of
    % fakeOptimization == 2 will be report
    
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
                % fprintf('Fold:%d,Iteration:%d, ObjectiveScore:%g\n', fold, iter, newObjectiveScore);
                oldObjectiveScore = newObjectiveScore;
                tmpOldObj=oldObjectiveScore;
                for dom = 1:numDom
                    [A,sumFi,E] = projectTensorToMatrix({CP1,CP2,CP3,CP4}, dom);
                    projB = A*sumFi*E';

                    if dom == targetDomain
                        V{dom} = V{dom}.*sqrt(((YMatrix{dom}.*W)'*U{dom}*projB)./(V{dom}*V{dom}'*(V{dom}*projB'*U{dom}'.*W')*U{dom}*projB));
                    else
                        V{dom} = V{dom}.*sqrt((YMatrix{dom}'*U{dom}*projB)./(V{dom}*V{dom}'*(V{dom}*projB'*U{dom}')*U{dom}*projB));
                    end                    
                    V{dom}(isnan(V{dom})) = 0;
                    V{dom}(~isfinite(V{dom})) = 0;                    
%                     tmpObjectiveScore = ShowObjectiveS(SU,SV,U, V, W, YMatrix, Lu, CP1, CP2, CP3, CP4, lambda);
%                     if tmpObjectiveScore > tmpOldObj
%                         fprintf('Domain:%d, Objective increased when update V (%f=>%f)\n', dom, tmpOldObj, tmpObjectiveScore);
%                     end
%                     tmpOldObj=tmpObjectiveScore;
                    %update U
                    if dom == targetDomain
                        U{dom} = U{dom}.*sqrt(((YMatrix{dom}.*W)*V{dom}*projB'+lambda*Su{dom}*U{dom})./(U{dom}*U{dom}'*(U{dom}*projB*V{dom}'.*W)*V{dom}*projB'+lambda*Du{dom}*U{dom}));
                    else
                        U{dom} = U{dom}.*sqrt((YMatrix{dom}*V{dom}*projB'+lambda*Su{dom}*U{dom})./(U{dom}*U{dom}'*U{dom}*projB*V{dom}'*V{dom}*projB'+lambda*Du{dom}*U{dom}));
                    end
                    U{dom}(isnan(U{dom})) = 0;
                    U{dom}(~isfinite(U{dom})) = 0;                    
%                     tmpObjectiveScore = ShowObjectiveS(SU,SV,U, V, W, YMatrix, Lu, CP1, CP2, CP3, CP4, lambda);
%                     if tmpObjectiveScore > tmpOldObj
%                         fprintf('Domain:%d, Objective increased when update U (%f=>%f)\n', dom, tmpOldObj, tmpObjectiveScore);
%                     end
%                     tmpOldObj=tmpObjectiveScore;
                    %update AE       
                    
                    if dom == sourceDomain
                        A = CP1;
                        E = CP2;
                    else
                        A = CP3;
                        E = CP4;
                    end
                    
                    [rA, cA] = size(A);
                    [rE, cE] = size(E);
                    
                    if dom ==targetDomain
                        A = A.*sqrt((U{dom}'*(YMatrix{dom}.*W)*V{dom}*E*sumFi)./(U{dom}'*(U{dom}*A*sumFi*E'*V{dom}'.*W)*V{dom}*E*sumFi+delta*ones(rE,rA)'*(sumFi*E')'));
                    else
                        A = A.*sqrt((U{dom}'*YMatrix{dom}*V{dom}*E*sumFi)./(U{dom}'*U{dom}*A*sumFi*E'*V{dom}'*V{dom}*E*sumFi+delta*ones(rE,rA)'*(sumFi*E')'));
                    end
                    A(isnan(A)) = 0;
                    A(~isfinite(A)) = 0;
                    if dom == sourceDomain
                        CP1 = A;
                    else
                        CP3 = A;
                    end
%                     tmpObjectiveScore = ShowObjectiveS(SU,SV,U, V, W, YMatrix, Lu, CP1, CP2, CP3, CP4, lambda);
%                     if tmpObjectiveScore > tmpOldObj
%                         fprintf('Domain:%d, Objective increased when update A (%f=>%f)\n', dom, tmpOldObj, tmpObjectiveScore);
%                     end
%                     tmpOldObj = tmpObjectiveScore;
                    if dom == targetDomain
                        E = E.*sqrt((V{dom}'*(YMatrix{dom}.*W)'*U{dom}*A*sumFi)./(V{dom}'*(V{dom}*E*sumFi*A'*U{dom}'.*W')*U{dom}*A*sumFi+delta*ones(rE,rA)*A*sumFi));
                    else
                        E = E.*sqrt((V{dom}'*YMatrix{dom}'*U{dom}*A*sumFi)./(V{dom}'*V{dom}*E*sumFi*A'*U{dom}'*U{dom}*A*sumFi+delta*ones(rE,rA)*A*sumFi));
                    end
                    E(isnan(E)) = 0;
                    E(~isfinite(E)) = 0;
                    if dom == sourceDomain
                        CP2 = E;
                    else
                        CP4 = E;
                    end
%                     tmpObjectiveScore = ShowObjectiveS(SU,SV,U, V, W, YMatrix, Lu, CP1, CP2, CP3, CP4, lambda);
%                     if tmpObjectiveScore > tmpOldObj
%                         fprintf('Domain:%d, Objective increased when update E (%f=>%f)\n', dom, tmpOldObj, tmpObjectiveScore);
%                     end
%                     tmpOldObj=tmpObjectiveScore;
                end
                newObjectiveScore = ShowObjectiveS(SU,SV,U, V, W, YMatrix, Lu, CP1, CP2, CP3, CP4, lambda);
                diff = oldObjectiveScore - newObjectiveScore;
            end
            disp(iter);
            foldObjectiveScores(fold) = newObjectiveScore;
            %calculate validationScore
            [A,sumFi,E] = projectTensorToMatrix({CP1,CP2,CP3,CP4}, targetDomain);
            projB = A*sumFi*E';
            result = U{targetDomain}*projB*V{targetDomain}';
            [~, maxIndex] = max(result, [], 2);
            predictResult = maxIndex;
            for dom = 1: CVFoldSize
                if(predictResult(validateIndex(dom)) == Label{targetDomain}(validateIndex(dom)))
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
            end
            time = round(clock);
            fprintf('Time: %d/%d/%d,%d:%d:%d\n', time(1), time(2), time(3), time(4), time(5), time(6));
            fprintf('Initial try: %d, ObjectiveScore:%f, Accuracy:%f%%\n', t, avgObjectiveScore, accuracy*100);
        end
    end
end

fprintf(resultFile, '%g,%g,%g,%g,%g,%g,%g\n', sigma, sigma2, lambda, delta, bestObjectiveScore, bestAccuracy, bestTime);
%     csvwrite(sprintf('../exp_result/predict_result/%s_predict_result.csv', exp_title), bestPredictResult);
