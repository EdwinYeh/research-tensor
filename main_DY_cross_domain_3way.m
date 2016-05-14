time = round(clock);
fprintf('Start Time: %d/%d/%d,%d:%d:%d\n', time(1), time(2), time(3), time(4), time(5), time(6));
fprintf('Use (cpRank, instanceCluster, featureCluster, Sigma, Lambda, Delta):(%g,%g,%g,%g,%g,%g)\n', cpRank, numInstanceCluster, numFeatureCluster, sigma, lambda, delta);

bestObjectiveScore = Inf;
bestTestAccuracy = 0;
bestTime = Inf;
SU=cell(1,2);
SV=cell(1,2);
U = cell(numCVFold,2);
V = cell(numCVFold,2);
CP1 = cell(numCVFold,1);
CP2 = cell(numCVFold,1);
CP3 = cell(numCVFold,1);
%CP4 = cell(numCVFold,1);

validationAccuracyList = zeros(randomTryTime, 1);
objectiveScoreList = zeros(randomTryTime, 1);
timeList = zeros(randomTryTime, 1);
for t = 1: randomTryTime
    
    for fold = 1: numCVFold
        CP1{fold} = rand(numInstanceCluster, cpRank);
        CP2{fold} = rand(numFeatureCluster, cpRank);
        CP3{fold} = rand(numInstanceCluster, cpRank);
        %CP4{fold} = rand(numFeatureCluster, cpRank);
        
        for dom = 1: 2
            U{fold, dom} = rand(numSampleInstance(dom), numInstanceCluster);
            V{fold, dom} = rand(2, numFeatureCluster);
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
        YMatrix = TrueYMatrix;
        W = ones(numSampleInstance(targetDomain), numClass(1));
        W(hiddenIndex, :) = 0;
        [rY,cY]=size(YMatrix{1});
        iter = 0;
        diff = Inf;
        newObjectiveScore = Inf;
        stopTag = 0;
        time = round(clock);
        fprintf('Initial time: %d/%d/%d,%d:%d:%d\n', time(1), time(2), time(3), time(4), time(5), time(6));
        while (stopTag < 50 && iter < maxIter)
            iter = iter + 1;
            %                 disp(diff);
            %             fprintf('Fold:%d,Iteration:%d, ObjectiveScore:%g\n',fold, iter, newObjectiveScore);
            oldObjectiveScore = newObjectiveScore;
            tmpOldObj=oldObjectiveScore;
            for dom = 1:numDom
                %[A,sumFi,E] = projectTensorToMatrix({CP1{fold},CP2{fold},CP3{fold},CP4{fold}}, dom);
                A = CP1{fold};
                if dom == sourceDomain
                    M=CP2{fold};
                    E = CP3{fold};
                else
                    M=CP3{fold};
                    E = CP2{fold};
                end
                [c,r]=size(M);
                psi=zeros(r);
                for i=1:c
                    psi=psi+diag(M(i,:));
                end
                sumFi=psi;
                
                projB = A*sumFi*E';
                
                if dom == targetDomain
                    V{fold,dom} = V{fold,dom}.*sqrt(((YMatrix{dom}.*W)'*U{fold,dom}*projB)./(V{fold,dom}*V{fold,dom}'*(YMatrix{dom}'.*W')*U{fold,dom}*projB));
                else
                    V{fold,dom} = V{fold,dom}.*sqrt((YMatrix{dom}'*U{fold,dom}*projB)./(V{fold,dom}*V{fold,dom}'*(YMatrix{dom}')*U{fold,dom}*projB));
                end
                
                %update U
                if dom == targetDomain
                    U{fold,dom} = U{fold,dom}.*sqrt(((YMatrix{dom}.*W)*V{fold,dom}*projB'+lambda*Su{dom}*U{fold,dom})./(U{fold,dom}*U{fold,dom}'*(YMatrix{dom}.*W)*V{fold,dom}*projB'+lambda*Du{dom}*U{fold,dom}));
                else
                    U{fold,dom} = U{fold,dom}.*sqrt((YMatrix{dom}*V{fold,dom}*projB'+lambda*Su{dom}*U{fold,dom})./(U{fold,dom}*U{fold,dom}'*YMatrix{dom}*V{fold,dom}*projB'+lambda*Du{dom}*U{fold,dom}));
                end
                
                %update AE
                
                
                [rA, cA] = size(A);
                [rE, cE] = size(E);
                %[Num,Den]=cross_domain_term(YMatrix,W,U,V,CP1,CP2,CP3,CP4,dom,fold,'A',delta);
                
                nu=(U{fold,2}'*(YMatrix{2}.*W)*V{fold,2}*CP2{fold}*sumFi);
                de=(U{fold,2}'*(U{fold,2}*A*sumFi*E'*V{fold,2}'.*W)*V{fold,2}*E*sumFi+delta*ones(rE,rA)'*(sumFi*E')');
                nu=nu+(U{fold,1}'*YMatrix{1}*V{fold,1}*CP3{fold}*sumFi);
                de=de+(U{fold,1}'*U{fold,1}*A*sumFi*CP3{fold}'*V{fold,1}'*V{fold,1}*CP3{fold}*sumFi+delta*ones(rE,rA)'*(sumFi*CP3{fold}')');
                A = A.*sqrt(nu./de);
                CP1{fold} = A;

                [Num,Den]=cross_domain_term_3way(YMatrix,W,U,V,CP1,CP2,CP3,dom,fold,delta);
                if dom == targetDomain
                    E = E.*sqrt((V{fold,dom}'*(YMatrix{dom}.*W)'*U{fold,dom}*A*sumFi+Num)./(V{fold,dom}'*(V{fold,dom}*E*sumFi*A'*U{fold,dom}'.*W')*U{fold,dom}*A*sumFi+delta*ones(rE,rA)*A*sumFi+Den));
                else
                    E = E.*sqrt((V{fold,dom}'*YMatrix{dom}'*U{fold,dom}*A*sumFi+Num)./(V{fold,dom}'*V{fold,dom}*E*sumFi*A'*U{fold,dom}'*U{fold,dom}*A*sumFi+delta*ones(rE,rA)*A*sumFi+Den));
                end
                %                     E(isnan(E)) = 0;
                %                     E(~isfinite(E)) = 0;
                if dom == sourceDomain
                    CP2{fold} = E;
                else
                    CP3{fold} = E;
                end
                
            end
            newObjectiveScore = ShowObjective_3way(fold, U, V, W, YMatrix, Lu, CP1, CP2, CP3, lambda, delta);
            objTrack{fold} = [objTrack{fold}, newObjectiveScore];
            diff = oldObjectiveScore - newObjectiveScore;
            if diff < 0.01
                stopTag = stopTag + 1;
            else
                stopTag = 0;
            end
        end
        time = round(clock);
        fprintf('Update time: %d/%d/%d,%d:%d:%d\n', time(1), time(2), time(3), time(4), time(5), time(6));
        foldObjectiveScores(fold) = newObjectiveScore;
        %Calculate objectiveScore
        %[A,sumFi,E] = projectTensorToMatrix({CP1{fold},CP2{fold},CP3{fold},CP4{fold}}, targetDomain);
% 	                A = CP1{fold};
%                 if dom == sourceDomain
%                     M=CP2{fold};
%                     E = CP3{fold};
%                 else
%                     M=CP3{fold};
%                     E = CP2{fold};
%                 end
%                 [c,r]=size(M);
%                 psi=zeros(r);
%                 for i=1:c
%                     psi=psi+diag(M(i,:));
%                 end
%                 sumFi=psi;


	
        projB = A*sumFi*E';
        result = U{fold,targetDomain}*projB*V{fold,targetDomain}';
        [~, maxIndex] = max(result, [], 2);
        predictResult = maxIndex;
        for dom = 1: CVFoldSize
            if(predictResult(hiddenIndex(dom)) == Label{targetDomain}(hiddenIndex(dom)))
                numCorrectPredict = numCorrectPredict + 1;
            end
        end
        hiddenIndex = hiddenIndex + CVFoldSize;
       	if fold == numCVFold && t == randomTryTime
            if delta==0
                save('full_3way_result_delta0.mat', 'U', 'V', 'CP1', 'CP2', 'CP3', 'objTrack');
            else
                save('full_3way_result_deltaNot0.mat', 'U', 'V', 'CP1', 'CP2', 'CP3', 'objTrack');
            end
        end
    end
    
    if isTestPhase
        accuracy = numCorrectPredict/ numTestInstance;
    else
        accuracy = numCorrectPredict/ numValidationInstance;
    end
    
    avgObjectiveScore = sum(foldObjectiveScores)/ numCVFold;
    avgTime = toc(TotalTimer)/ numCVFold;
    time = round(clock);
    fprintf('PredictionTime: %d/%d/%d,%d:%d:%d\n', time(1), time(2), time(3), time(4), time(5), time(6));
    if isTestPhase
        fprintf(resultFile, '%g,%g,%g,%g,%g,%g,%g,%g,%g\n', cpRank, numInstanceCluster, numFeatureCluster, sigma, lambda, delta, avgObjectiveScore, accuracy, avgTime);
        if accuracy > bestTestAccuracy
            bestObjectiveScore = avgObjectiveScore;
            bestTestAccuracy = accuracy;
            bestTime = avgTime;
        end
    end
    time = round(clock);
    validationAccuracyList(t) = accuracy;
    objectiveScoreList(t) = avgObjectiveScore;
    timeList(t) = avgTime;
    %     fprintf('randomTime:%d, accuracy: %g, objectiveScore:%g\n', t, accuracy, avgObjectiveScore);
end

if isTestPhase
%     fprintf(resultFile, '%g,%g,%g,%g,%g,%g,%g,%g,%g\n', cpRank, numInstanceCluster, numFeatureCluster, sigma, lambda, delta, bestObjectiveScore, bestTestAccuracy, bestTime);
else
    avgValidationAccuracy = sum(validationAccuracyList)/ randomTryTime;
    avgObjectiveScore = sum(objectiveScoreList)/ randomTryTime;
    avgTime = sum(timeList)/ randomTryTime;
    fprintf('avgValidationAccuracy: %g, objectiveScore:%g\n', avgValidationAccuracy, avgObjectiveScore);
    compareWithTheBest(avgValidationAccuracy, avgObjectiveScore, avgTime, sigma, lambda, delta, cpRank, numInstanceCluster, numFeatureCluster, resultDirectory, expTitle)
    fprintf(resultFile, '%g,%g,%g,%g,%g,%g,%g,%g,%g\n', cpRank, numInstanceCluster, numFeatureCluster, sigma, lambda, delta, avgObjectiveScore, avgValidationAccuracy, avgTime);
end
time = round(clock);
fprintf('OtherTime: %d/%d/%d,%d:%d:%d\n', time(1), time(2), time(3), time(4), time(5), time(6));