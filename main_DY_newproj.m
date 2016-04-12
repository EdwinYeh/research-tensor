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
SU=cell(1,2);
SV=cell(1,2);
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
    U{1}=diag(1./(sum(U{1},2)+eps))*U{1};
    U{2}=diag(1./(sum(U{2},2)+eps))*U{2};

    V = initV(t, :);
    V{1}=diag(1./(sum(V{1},2)+eps))*V{1};
    V{2}=diag(1./(sum(V{2},2)+eps))*V{2};

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
                
%                 V{i} = V{i}.*sqrt((YMatrix{i}'*U{i}*projB)./((V{i}*projB'*U{i}')*U{i}*projB));
                if i == targetDomain
                    V{i} = V{i}.*sqrt(((YMatrix{i}.*W)'*U{i}*projB)./(V{i}*V{i}'*(V{i}*projB'*U{i}'.*W')*U{i}*projB));
                else
                    V{i} = V{i}.*sqrt((YMatrix{i}'*U{i}*projB)./(V{i}*V{i}'*(V{i}*projB'*U{i}')*U{i}*projB));
                end
                V{i}(isnan(V{i})) = 0;
                V{i}(~isfinite(V{i})) = 0;
                
                %col normalize
                %[r, ~] = size(V{i});
                %for tmpI = 1:r
	        %    bot = sum(abs(V{i}(tmpI,:)));
                %    if bot == 0
                %        bot = 1;
                %    end
                %    V{i}(tmpI,:) = V{i}(tmpI,:)/bot;
                %end
		
		%S=diag(1./(sum(V{i},2)+eps));
		%V{i}=S*V{i};
		%SV{i}=SV{i}/S;

                %V{i}(isnan(V{i})) = 0;
                %V{i}(~isfinite(V{i})) = 0;
                tmpObjectiveScore = ShowObjectiveS(SU,SV,U, V, W, YMatrix, Lu, CP1, CP2, CP3, CP4, lambda);
                if tmpObjectiveScore > tmpOldObj
                    fprintf('Objective increased when update V (%f=>%f)\n', tmpOldObj, tmpObjectiveScore);
                end
		tmpOldObj=tmpObjectiveScore;
                %update U
%                 U{i} = U{i}.*sqrt((YMatrix{i}*V{i}*projB'+lambda*Su{i}*U{i})./(U{i}*projB*V{i}'*V{i}*projB'+lambda*Du{i}*U{i}));
                if i == targetDomain
                    U{i} = U{i}.*sqrt(((YMatrix{i}.*W)*V{i}*projB'+lambda*Su{i}*U{i})./(U{i}*U{i}'*(U{i}*projB*V{i}'.*W)*V{i}*projB'+lambda*Du{i}*U{i}));
                else
                    U{i} = U{i}.*sqrt((YMatrix{i}*V{i}*projB'+lambda*Su{i}*U{i})./(U{i}*U{i}'*U{i}*projB*V{i}'*V{i}*projB'+lambda*Du{i}*U{i}));
                end
                U{i}(isnan(U{i})) = 0;
                U{i}(~isfinite(U{i})) = 0;
                 
                %col normalize
                %[r, ~] = size(U{i});
                %for tmpI = 1:r
                %    bot = sum(abs(U{i}(tmpI,:)));
                %    if bot == 0
                %        bot = 1;
                %    end
                %    U{i}(tmpI,:) = U{i}(tmpI,:)/bot;
                %end


		%S=diag(1./(sum(U{i},2)+eps));
		%U{i}=S*U{i};
		%SU{i}=SU{i}/S;

                %U{i}(isnan(U{i})) = 0;
                %U{i}(~isfinite(U{i})) = 0;
                 tmpObjectiveScore = ShowObjectiveS(SU,SV,U, V, W, YMatrix, Lu, CP1, CP2, CP3, CP4, lambda);
                if tmpObjectiveScore > tmpOldObj
                    fprintf('Objective increased when update U (%f=>%f)\n', tmpOldObj, tmpObjectiveScore);
                end
		tmpOldObj=tmpObjectiveScore;
                %update fi
                [rA, cA] = size(A);
                onesA = ones(rA, cA);
                A = A.*sqrt((U{i}'*YMatrix{i}*V{i}*E*sumFi)./(U{i}'*U{i}*A*sumFi*E'*V{i}'*V{i}*E*sumFi+0.0001*repmat(sum(sumFi*E',2)',[rA,1])));
                A(isnan(A)) = 0;
                A(~isfinite(A)) = 0;
                if i == sourceDomain
                    CP1 = A;
                else
                    CP3 = A;
                end
                 tmpObjectiveScore = ShowObjectiveS(SU,SV,U, V, W, YMatrix, Lu, CP1, CP2, CP3, CP4, lambda);
                if tmpObjectiveScore > tmpOldObj
                    fprintf('Objective increased when update A (%f=>%f)\n', tmpOldObj, tmpObjectiveScore);
                end
		tmpOldObj = oldObjectiveScore;

                [rE ,cE] = size(E);
                onesE = ones(rE, cE);
                E = E.*sqrt((V{i}'*YMatrix{i}'*U{i}*A*sumFi)./(V{i}'*V{i}*E*sumFi*A'*U{i}'*U{i}*A*sumFi+0.0001*repmat(sum(A*sumFi,1),[rE,1])));
                E(isnan(E)) = 0;
                E(~isfinite(E)) = 0;
                if i == sourceDomain
                   CP2 = E;
                else
                   CP4 = E;
               end
                tmpObjectiveScore = ShowObjectiveS(SU,SV,U, V, W, YMatrix, Lu, CP1, CP2, CP3, CP4, lambda);
                if tmpObjectiveScore > tmpOldObj
                    fprintf('Objective increased when update E (%f=>%f)\n', tmpOldObj, tmpObjectiveScore);
                end
		tmpOldObj=tmpObjectiveScore;
		
            end
            %for i = 1:numDom
            %    [A,sumFi,E] = projectTensorToMatrix({CP1,CP2,CP3,CP4}, i);
            %    projB = A*sumFi*E';
            %    result = U{i}*projB*V{i}';
            %    if i == targetDomain
            %        normEmp = norm((YMatrix{i} - result).*W, 'fro')*norm((YMatrix{i} - result).*W, 'fro');
            %    else
            %        normEmp = norm((YMatrix{i} - result), 'fro')*norm((YMatrix{i} - result), 'fro');
            %    end
            %    smoothU = lambda*trace(U{i}'*Lu{i}*U{i});
            %    objectiveScore = normEmp + smoothU;
            %    newObjectiveScore = newObjectiveScore + objectiveScore;
            %end
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
