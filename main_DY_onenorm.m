time = round(clock);
fprintf('Time: %d/%d/%d,%d:%d:%d\n', time(1), time(2), time(3), time(4), time(5), time(6));
fprintf('Use (Sigma, Sigma2, Lambda, Delta):(%g,%g,%g,%g)\n', sigma, sigma2, lambda, delta);

bestObjectiveScore = Inf;
bestAccuracy = 0;
bestTime = 0;
SU=cell(1,2);
SV=cell(1,2);
U = cell(randomTryTime,numCVFold,2,1);
V = cell(randomTryTime,numCVFold,2,1);
realU = cell(randomTryTime,numCVFold,2,1);
realV = cell(randomTryTime,numCVFold,2,1);
CP1 = cell(randomTryTime,numCVFold);
CP2 = cell(randomTryTime,numCVFold);
CP3 = cell(randomTryTime,numCVFold);
CP4 = cell(randomTryTime,numCVFold);
realCP1 = cell(randomTryTime,numCVFold);
realCP2 = cell(randomTryTime,numCVFold);
realCP3 = cell(randomTryTime,numCVFold);
realCP4 = cell(randomTryTime,numCVFold);

for t = 1: randomTryTime    
    U = cell(numCVFold, 2);
    V = cell(numCVFold, 2);
    tmpU = cell(numCVFold, 2);
    tmpV = cell(numCVFold, 2);
    realU = cell(numCVFold, 2);
    realV = cell(numCVFold, 2);
    CP1 = cell(numCVFold, 1);
    CP2 = cell(numCVFold, 1);
    CP3 = cell(numCVFold, 1);
    CP4 = cell(numCVFold, 1);
    
    for fold = 1: numCVFold
        CP1{fold} = rand(numInstanceCluster, cpRank);
        CP2{fold} = rand(numFeatureCluster, cpRank);
        CP3{fold} = rand(numInstanceCluster, cpRank);
        CP4{fold} = rand(numFeatureCluster, cpRank);
        
        for dom = 1: 2
            U{t, fold, dom} = rand(numSampleInstance(dom), numInstanceCluster);
            V{t, fold, dom} = rand(2, numFeatureCluster);
        end
    end
end

for t = 1: randomTryTime
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
            SU{1} = eye(rY);
            SU{2} = SU{1};
            SV{1} = eye(cY);
            SV{2} = SV{1};
            iter = 0;
            diff = Inf;
            newObjectiveScore = Inf;
            
            while ((fakeOptimization == 2 && diff >= 0.0001 && iter < maxIter)||(fakeOptimization ~= 2 && iter < maxIter))
                iter = iter + 1;
                fprintf('Fake:%d, Fold:%d, Iteration:%d, ObjectiveScore:%g\n', fakeOptimization, fold, iter, newObjectiveScore);
                oldObjectiveScore = newObjectiveScore;
                tmpOldObj=oldObjectiveScore;
                for dom = 1:numDom
                    [A,sumFi,E] = projectTensorToMatrix({CP1{t,fold},CP2{t,fold},CP3{t,fold},CP4{t,fold}}, dom);
                    projB = A*sumFi*E';
                    
                    if dom == targetDomain
                        V{t,fold,dom} = V{t,fold,dom}.*sqrt(((YMatrix{dom}.*W)'*U{t,fold,dom}*projB)./(V{t,fold,dom}*V{t,fold,dom}'*(V{t,fold,dom}*projB'*U{t,fold,dom}'.*W')*U{t,fold,dom}*projB));
                    else
                        V{t,fold,dom} = V{t,fold,dom}.*sqrt((YMatrix{dom}'*U{t,fold,dom}*projB)./(V{t,fold,dom}*V{t,fold,dom}'*(V{t,fold,dom}*projB'*U{t,fold,dom}')*U{t,fold,dom}*projB));
                    end
                    V{t,fold,dom}(isnan(V{t,fold,dom})) = 0;
                    V{t,fold,dom}(~isfinite(V{t,fold,dom})) = 0;
                    %                     tmpObjectiveScore = ShowObjectiveS(SU,SV,U, V, W, YMatrix, Lu, CP1{t,fold}, CP2{t,fold}, CP3{t,fold}, CP4{t,fold}, lambda);
                    %                     if tmpObjectiveScore > tmpOldObj
                    %                         fprintf('Domain:%d, Objective increased when update V (%f=>%f)\n', dom, tmpOldObj, tmpObjectiveScore);
                    %                     end
                    %                     tmpOldObj=tmpObjectiveScore;
                    %update U
                    if dom == targetDomain
                        U{t,fold,dom} = U{t,fold,dom}.*sqrt(((YMatrix{dom}.*W)*V{t,fold,dom}*projB'+lambda*Su{dom}*U{t,fold,dom})./(U{t,fold,dom}*U{t,fold,dom}'*(U{t,fold,dom}*projB*V{t,fold,dom}'.*W)*V{t,fold,dom}*projB'+lambda*Du{dom}*U{t,fold,dom}));
                    else
                        U{t,fold,dom} = U{t,fold,dom}.*sqrt((YMatrix{dom}*V{t,fold,dom}*projB'+lambda*Su{dom}*U{t,fold,dom})./(U{t,fold,dom}*U{t,fold,dom}'*U{t,fold,dom}*projB*V{t,fold,dom}'*V{t,fold,dom}*projB'+lambda*Du{dom}*U{t,fold,dom}));
                    end
                    U{t,fold,dom}(isnan(U{t,fold,dom})) = 0;
                    U{t,fold,dom}(~isfinite(U{t,fold,dom})) = 0;
                    %                     tmpObjectiveScore = ShowObjectiveS(SU,SV,U, V, W, YMatrix, Lu, CP1{t,fold}, CP2{t,fold}, CP3{t,fold}, CP4{t,fold}, lambda);
                    %                     if tmpObjectiveScore > tmpOldObj
                    %                         fprintf('Domain:%d, Objective increased when update U (%f=>%f)\n', dom, tmpOldObj, tmpObjectiveScore);
                    %                     end
                    %                     tmpOldObj=tmpObjectiveScore;
                    %update AE
                    if dom == sourceDomain
                        A = CP1{t,fold};
                        E = CP2{t,fold};
                    else
                        A = CP3{t,fold};
                        E = CP4{t,fold};
                    end
                    
                    [rA, cA] = size(A);
                    [rE, cE] = size(E);
                    
                    if dom ==targetDomain
                        A = A.*sqrt((U{t,fold,dom}'*(YMatrix{dom}.*W)*V{t,fold,dom}*E*sumFi)./(U{t,fold,dom}'*(U{t,fold,dom}*A*sumFi*E'*V{t,fold,dom}'.*W)*V{t,fold,dom}*E*sumFi+delta*ones(rE,rA)'*(sumFi*E')'));
>>>>>>> parent of 503cc95... [correct version]
                    else
                        A = A.*sqrt((U{t,fold,dom}'*YMatrix{dom}*V{t,fold,dom}*E*sumFi)./(U{t,fold,dom}'*U{t,fold,dom}*A*sumFi*E'*V{t,fold,dom}'*V{t,fold,dom}*E*sumFi+delta*ones(rE,rA)'*(sumFi*E')'));
                    end
                    A(isnan(A)) = 0;
                    A(~isfinite(A)) = 0;
                    if dom == sourceDomain
<<<<<<< HEAD
                        tmpCP1{fold} = A;              
                        tmpObjectiveScore = ShowObjective(fold, U, V, W, YMatrix, Lu, tmpCP1, CP2, CP3, CP4, lambda);
                    else
                        tmpCP3{fold} = A;
                        tmpObjectiveScore = ShowObjective(fold, U, V, W, YMatrix, Lu, CP1, CP2, tmpCP3, CP4, lambda);
                    end
                    if fakeOptimization == 2
                        if tmpObjectiveScore < tmpOldObj
                            fprintf('Update A (%f=>%f)\n', tmpOldObj, tmpObjectiveScore);
                            if dom == sourceDomain
                                CP1 = tmpCP1;
                            else
                                CP3 = tmpCP3;
                            end
                        else
                            tmpCP1 = CP1;
                            tmpCP3 = CP3;
                            fprintf('Did not update A (%f=>%f)\n', tmpOldObj, tmpObjectiveScore);
                        end
                    else
                        if dom == sourceDomain
                            CP1 = tmpCP1;
                        else
                            CP3 = tmpCP3;
                        end
                        tmpOldObj = tmpObjectiveScore;
                    end                   
                    
=======
                        CP1{t,fold} = A;
                    else
                        CP3{t,fold} = A;
                    end
                    %                     tmpObjectiveScore = ShowObjectiveS(SU,SV,U, V, W, YMatrix, Lu, CP1{t,fold}, CP2{t,fold}, CP3{t,fold}, CP4{t,fold}, lambda);
                    %                     if tmpObjectiveScore > tmpOldObj
                    %                         fprintf('Domain:%d, Objective increased when update A (%f=>%f)\n', dom, tmpOldObj, tmpObjectiveScore);
                    %                     end
                    %                     tmpOldObj = tmpObjectiveScore;
>>>>>>> parent of 503cc95... [correct version]
                    if dom == targetDomain
                        E = E.*sqrt((V{t,fold,dom}'*(YMatrix{dom}.*W)'*U{t,fold,dom}*A*sumFi)./(V{t,fold,dom}'*(V{t,fold,dom}*E*sumFi*A'*U{t,fold,dom}'.*W')*U{t,fold,dom}*A*sumFi+delta*ones(rE,rA)*A*sumFi));
                    else
                        E = E.*sqrt((V{t,fold,dom}'*YMatrix{dom}'*U{t,fold,dom}*A*sumFi)./(V{t,fold,dom}'*V{t,fold,dom}*E*sumFi*A'*U{t,fold,dom}'*U{t,fold,dom}*A*sumFi+delta*ones(rE,rA)*A*sumFi));
                    end
                    E(isnan(E)) = 0;
                    E(~isfinite(E)) = 0;
                    if dom == sourceDomain
<<<<<<< HEAD
                        tmpCP2{fold} = E;              
                        tmpObjectiveScore = ShowObjective(fold, U, V, W, YMatrix, Lu, CP1, tmpCP2, CP3, CP4, lambda);
                    else
                        tmpCP4{fold} = E;
                        tmpObjectiveScore = ShowObjective(fold, U, V, W, YMatrix, Lu, CP1, CP2, CP3, tmpCP4, lambda);
                    end
                    if fakeOptimization == 2
                        if tmpObjectiveScore < tmpOldObj
                            if dom == sourceDomain
                                CP2 = tmpCP2;
                            else
                                CP4 = tmpCP4;
                            end
                            tmpOldObj = tmpObjectiveScore;
                            fprintf('Update E (%f=>%f)\n', tmpOldObj, tmpObjectiveScore);
                        else
                            tmpCP2 = CP2;
                            tmpCP4 = CP4;
                            fprintf('Did not update E (%f=>%f)\n', tmpOldObj, tmpObjectiveScore);
                        end   
                    else
                        if dom == sourceDomain
                            CP2 = tmpCP2;
                        else
                            CP4 = tmpCP4;
                        end
                        tmpOldObj = tmpObjectiveScore;
                    end
                    
=======
                        CP2{t,fold} = E;
                    else
                        CP4{t,fold} = E;
                    end
                    %                     tmpObjectiveScore = ShowObjectiveS(SU,SV,U, V, W, YMatrix, Lu, CP1{t,fold}, CP2{t,fold}, CP3{t,fold}, CP4{t,fold}, lambda);
                    %                     if tmpObjectiveScore > tmpOldObj
                    %                         fprintf('Domain:%d, Objective increased when update E (%f=>%f)\n', dom, tmpOldObj, tmpObjectiveScore);
                    %                     end
                    %                     tmpOldObj=tmpObjectiveScore;
>>>>>>> parent of 503cc95... [correct version]
                end
                newObjectiveScore = ShowObjective(U, V, W, YMatrix, Lu, CP1{t,fold}, CP2{t,fold}, CP3{t,fold}, CP4{t,fold}, lambda);
                diff = oldObjectiveScore - newObjectiveScore;
            end
            foldObjectiveScores(fold) = newObjectiveScore;
            %calculate validationScore
            [A,sumFi,E] = projectTensorToMatrix({CP1{t,fold},CP2{t,fold},CP3{t,fold},CP4{t,fold}}, targetDomain);
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
            
            if fakeOptimization == 1
                realCP1{t, fold} = CP1{t, fold};
                realCP2{t, fold} = CP2{t, fold};
                realCP3{t, fold} = CP3{t, fold};
                realCP4{t, fold} = CP4{t, fold};
                realU{t, fold} = U{t, fold};
                realV{t, fold} = V{t, fold};
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

fprintf(resultFile, '%g,%g,%g,%g,%g,%g,%g\n', sigma, sigma2, lambda, delta, bestObjectiveScore, bestAccuracy, bestTime);
%     csvwrite(sprintf('../exp_result/predict_result/%s_predict_result.csv', exp_title), bestPredictResult);