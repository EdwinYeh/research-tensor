time = round(clock);
fprintf('Time: %d/%d/%d,%d:%d:%d\n', time(1), time(2), time(3), time(4), time(5), time(6));
fprintf('Use (cpRank, instanceCluster, featureCluster, Sigma, Sigma2, Lambda, Delta):(%g,%g,%g,%g,%g,%g,%g)\n', cpRank, numInstanceCluster, numFeatureCluster, sigma, sigma2, lambda, delta);

bestObjectiveScore = Inf;
bestAccuracy = 0;
bestTime = 0;
SU=cell(1,2);
SV=cell(1,2);
U = cell(numCVFold,2);
V = cell(numCVFold,2);
realU = cell(numCVFold,2);
realV = cell(numCVFold,2);
CP1 = cell(numCVFold,1);
CP2 = cell(numCVFold,1);
CP3 = cell(numCVFold,1);
CP4 = cell(numCVFold,1);

for t = 1: randomTryTime
    
    for fold = 1: numCVFold
        CP1{fold} = rand(numInstanceCluster, cpRank);
        CP2{fold} = rand(numFeatureCluster, cpRank);
        CP3{fold} = rand(numInstanceCluster, cpRank);
        CP4{fold} = rand(numFeatureCluster, cpRank);
        
        for dom = 1: 2
            U{fold, dom} = rand(numSampleInstance(dom), numInstanceCluster);
            V{fold, dom} = rand(2, numFeatureCluster);
        end
    end
    % When fakeOptimization == 1, train UVAE  to be the initial points
    % during fakeOptimization == 2. Only the report the result of
    % fakeOptimization == 2 will be report
    
    for fakeOptimization = 1: 2
        numCorrectPredict = 0;
        validateIndex = 1: CVFoldSize;
        TotalTimer = tic;
        foldObjectiveScores = zeros(1,numCVFold);
        
        for fold = 1:numCVFold
            YMatrix = TrueYMatrix;
            YMatrix{targetDomain}(validateIndex, :) = zeros(CVFoldSize, numClass(1));
            W = ones(numSampleInstance(targetDomain), numClass(1));
            W(validateIndex, :) = 0;
            [rY,cY]=size(YMatrix{1});
            iter = 0;
            diff = Inf;
            newObjectiveScore = Inf;
            
            if fakeOptimization == 2
                U = realU;
                V = realV;
                maxIter = 1000;
            else
                maxIter = 50;
            end
            
            while ((fakeOptimization==2 && diff>=0.001  && iter<maxIter)||(fakeOptimization~=2 && iter<maxIter))
                iter = iter + 1;
%                 disp(diff);
%                 fprintf('Fake:%d, Fold:%d,Iteration:%d, ObjectiveScore:%g\n', fakeOptimization, fold, iter, newObjectiveScore);
                oldObjectiveScore = newObjectiveScore;
                tmpOldObj=oldObjectiveScore;
                for dom = 1:numDom
                    [A,sumFi,E] = projectTensorToMatrix({CP1{fold},CP2{fold},CP3{fold},CP4{fold}}, dom);
                    projB = A*sumFi*E';
                    
                    if dom == targetDomain
                        V{fold,dom} = V{fold,dom}.*sqrt(((YMatrix{dom}.*W)'*U{fold,dom}*projB)./(V{fold,dom}*V{fold,dom}'*(YMatrix{dom}'.*W')*U{fold,dom}*projB));
                    else
                        V{fold,dom} = V{fold,dom}.*sqrt((YMatrix{dom}'*U{fold,dom}*projB)./(V{fold,dom}*V{fold,dom}'*(YMatrix{dom}')*U{fold,dom}*projB));
                    end
                    V{fold,dom}(isnan(V{fold,dom})) = 0;
                    V{fold,dom}(~isfinite(V{fold,dom})) = 0;
                    %                     tmpObjectiveScore = ShowObjectiveS(SU,SV,U, V, W, YMatrix, Lu, CP1{fold}, CP2{fold}, CP3{fold}, CP4{fold}, lambda);
                    %                     if tmpObjectiveScore > tmpOldObj
                    %                         fprintf('Domain:%d, Objective increased when update V (%f=>%f)\n', dom, tmpOldObj, tmpObjectiveScore);
                    %                     end
                    %                     tmpOldObj=tmpObjectiveScore;
                    %update U
                    if dom == targetDomain
                        U{fold,dom} = U{fold,dom}.*sqrt(((YMatrix{dom}.*W)*V{fold,dom}*projB'+lambda*Su{dom}*U{fold,dom})./(U{fold,dom}*U{fold,dom}'*(YMatrix{dom}'.*W)*V{fold,dom}*projB'+lambda*Du{dom}*U{fold,dom}));
                    else
                        U{fold,dom} = U{fold,dom}.*sqrt((YMatrix{dom}*V{fold,dom}*projB'+lambda*Su{dom}*U{fold,dom})./(U{fold,dom}*U{fold,dom}'*YMatrix{dom}'*V{fold,dom}*projB'+lambda*Du{dom}*U{fold,dom}));
                    end
                    U{fold,dom}(isnan(U{fold,dom})) = 0;
                    U{fold,dom}(~isfinite(U{fold,dom})) = 0;
                    %                     tmpObjectiveScore = ShowObjectiveS(SU,SV,U, V, W, YMatrix, Lu, CP1{fold}, CP2{fold}, CP3{fold}, CP4{fold}, lambda);
                    %                     if tmpObjectiveScore > tmpOldObj
                    %                         fprintf('Domain:%d, Objective increased when update U (%f=>%f)\n', dom, tmpOldObj, tmpObjectiveScore);
                    %                     end
                    %                     tmpOldObj=tmpObjectiveScore;
                    %update AE
                    if dom == sourceDomain
                        A = CP1{fold};
                        E = CP2{fold};
                    else
                        A = CP3{fold};
                        E = CP4{fold};
                    end
                    
                    [rA, cA] = size(A);
                    [rE, cE] = size(E);
                    
                    if dom ==targetDomain
                        A = A.*sqrt((U{fold,dom}'*(YMatrix{dom}.*W)*V{fold,dom}*E*sumFi)./(U{fold,dom}'*(U{fold,dom}*A*sumFi*E'*V{fold,dom}'.*W)*V{fold,dom}*E*sumFi+delta*ones(rE,rA)'*(sumFi*E')'));
                    else
                        A = A.*sqrt((U{fold,dom}'*YMatrix{dom}*V{fold,dom}*E*sumFi)./(U{fold,dom}'*U{fold,dom}*A*sumFi*E'*V{fold,dom}'*V{fold,dom}*E*sumFi+delta*ones(rE,rA)'*(sumFi*E')'));
                    end
                    A(isnan(A)) = 0;
                    A(~isfinite(A)) = 0;
                    if dom == sourceDomain
                        CP1{fold} = A;
                    else
                        CP3{fold} = A;
                    end
                    %                     tmpObjectiveScore = ShowObjectiveS(SU,SV,U, V, W, YMatrix, Lu, CP1{fold}, CP2{fold}, CP3{fold}, CP4{fold}, lambda);
                    %                     if tmpObjectiveScore > tmpOldObj
                    %                         fprintf('Domain:%d, Objective increased when update A (%f=>%f)\n', dom, tmpOldObj, tmpObjectiveScore);
                    %                     end
                    %                     tmpOldObj = tmpObjectiveScore;
                    if dom == targetDomain
                        E = E.*sqrt((V{fold,dom}'*(YMatrix{dom}.*W)'*U{fold,dom}*A*sumFi)./(V{fold,dom}'*(V{fold,dom}*E*sumFi*A'*U{fold,dom}'.*W')*U{fold,dom}*A*sumFi+delta*ones(rE,rA)*A*sumFi));
                    else
                        E = E.*sqrt((V{fold,dom}'*YMatrix{dom}'*U{fold,dom}*A*sumFi)./(V{fold,dom}'*V{fold,dom}*E*sumFi*A'*U{fold,dom}'*U{fold,dom}*A*sumFi+delta*ones(rE,rA)*A*sumFi));
                    end
                    E(isnan(E)) = 0;
                    E(~isfinite(E)) = 0;
                    if dom == sourceDomain
                        CP2{fold} = E;
                    else
                        CP4{fold} = E;
                    end
                    %                     tmpObjectiveScore = ShowObjectiveS(SU,SV,U, V, W, YMatrix, Lu, CP1{fold}, CP2{fold}, CP3{fold}, CP4{fold}, lambda);
                    %                     if tmpObjectiveScore > tmpOldObj
                    %                         fprintf('Domain:%d, Objective increased when update E (%f=>%f)\n', dom, tmpOldObj, tmpObjectiveScore);
                    %                     end
                    %                     tmpOldObj=tmpObjectiveScore;
                end
                newObjectiveScore = ShowObjective(fold, U, V, W, YMatrix, Lu, CP1, CP2, CP3, CP4, lambda);
                diff = oldObjectiveScore - newObjectiveScore;
            end
            foldObjectiveScores(fold) = newObjectiveScore;
            %calculate validationScore
            [A,sumFi,E] = projectTensorToMatrix({CP1{fold},CP2{fold},CP3{fold},CP4{fold}}, targetDomain);
            projB = A*sumFi*E';
            result = U{fold,targetDomain}*projB*V{fold,targetDomain}';
            [~, maxIndex] = max(result, [], 2);
            predictResult = maxIndex;
            for dom = 1: CVFoldSize
                if(predictResult(validateIndex(dom)) == Label{targetDomain}(validateIndex(dom)))
                    numCorrectPredict = numCorrectPredict + 1;
                end
            end
            validateIndex = validateIndex + CVFoldSize;
            
            if fakeOptimization == 1
                realU = U;
                realV = V;
            end
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
            %             fprintf('Time: %d/%d/%d,%d:%d:%d\n', time(1), time(2), time(3), time(4), time(5), time(6));
            %             fprintf('Initial try: %d, ObjectiveScore:%f, Accuracy:%f%%\n', t, avgObjectiveScore, accuracy*100);
        end
    end
end

fprintf(resultFile, '%g,%g,%g,%g,%g,%g,%g,%g,%g,%g\n', cpRank, numInstanceCluster, numFeatureCluster, sigma, sigma2, lambda, delta, bestObjectiveScore, bestAccuracy, bestTime);
%     csvwrite(sprintf('../exp_result/predict_result/%s_predict_result.csv', exp_title), bestPredictResult);