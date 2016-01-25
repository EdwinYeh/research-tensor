resultFile = fopen(sprintf('../exp_result/result_%s.csv', exp_title), 'w');
fprintf(resultFile, 'lambda,objectiveScore,accuracy\n');
disp('Start training');

for tuneLambda = 0:lambdaTryTime
    lambda = 0.001 * 10 ^ tuneLambda;
    time = round(clock);
    fprintf('Time: %d/%d/%d,%d:%d:%d\n', time(1), time(2), time(3), time(4), time(5), time(6));
    fprintf('Use Lambda:%f\n', lambda);
    %each pair is (objective score, accuracy);
    resultCellArray = cell(randomTryTime, 3);
    for t = 1: randomTryTime
        numCorrectPredict = 0;
        validateIndex = 1: CVFoldSize;
        foldObjectiveScores = zeros(1,numCVFold);
        TotalTimer = tic;
        for fold = 1:numCVFold
            YMatrix = TrueYMatrix;
            YMatrix{targetDomain}(validateIndex, :) = zeros(CVFoldSize, numClass(1));
            W = ones(numSampleInstance(targetDomain), numClass(1));
            W(validateIndex, :) = 0;
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
                    [projB, threeMatrixB] = SumOfMatricize(B, 2*(i - 1)+1);
                    %bestCPR = FindBestRank(threeMatrixB, 50)
                    bestCPR = 20;
%                     cpTimer = tic;
                    CP = cp_apr(tensor(threeMatrixB), bestCPR, 'printitn', 0, 'alg', 'mu');%parafac_als(tensor(threeMatrixB), bestCPR);
%                     cpUsedTime = toc(cpTimer);
%                     disp(cpUsedTime);
                    A = CP.U{1};
                    E = CP.U{2};
                    U3 = CP.U{3};

                    fi = cell(1, length(CP.U{3}));
                    
%                     updateUVTimer = tic;
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
                    
                    %update U
                    if i == targetDomain
                        U{i} = U{i}.*sqrt((YMatrix{i}*V{i}*projB' + lambda*Su{i}*U{i})./((U{i}*projB*V{i}'.*W)*V{i}*projB' + lambda*Du{i}*U{i}));
                    else
                        U{i} = U{i}.*sqrt((YMatrix{i}*V{i}*projB' + lambda*Su{i}*U{i})./(U{i}*projB*V{i}'*V{i}*projB' + lambda*Du{i}*U{i}));
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
%                     updateUVTime = toc(updateUVTimer);
%                     disp(updateUVTime);

                    %update fi
                    [r, c] = size(U3);
                    nextThreeB = zeros(numInstanceCluster, numFeatureCluster, r);
                    sumFi = zeros(c, c);
                    CPLamda = CP.lambda(:);
                    for idx = 1:r
                        fi{idx} = diag(CPLamda.*U3(idx,:)');
                        sumFi = sumFi + fi{idx};
                    end
                    if isUpdateAE
%                         updateAETimer = tic;
                        [rA, cA] = size(A);
                        onesA = ones(rA, cA);
                        A = A.*sqrt((U{i}'*YMatrix{i}*V{i}*E*sumFi + alpha*(onesA))./(U{i}'*U{i}*A*sumFi*E'*V{i}'*V{i}*E*sumFi));
                        A(isnan(A)) = 0;
                        A(~isfinite(A)) = 0;
                        A(isnan(A)) = 0;
                        A(~isfinite(A)) = 0;

                        [rE ,cE] = size(E);
                        onesE = ones(rE, cE);
                        E = E.*sqrt((V{i}'*YMatrix{i}'*U{i}*A*sumFi + beta*(onesE))./(V{i}'*V{i}*E*sumFi*A'*U{i}'*U{i}*A*sumFi));
                        E(isnan(E)) = 0;
                        E(~isfinite(E)) = 0;
                        E(isnan(E)) = 0;
                        E(~isfinite(E)) = 0;
%                         updateAETime = toc(updateAETimer);
%                         disp(updateAETime);
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
                        normEmp = norm((YMatrix{i} - result).*W)*norm((YMatrix{i} - result).*W);
                    else
                        normEmp = norm((YMatrix{i} - result))*norm((YMatrix{i} - result));
                    end
                    smoothU = lambda*trace(U{i}'*Lu{i}*U{i});
                    objectiveScore = normEmp + smoothU;
                    newObjectiveScore = newObjectiveScore + objectiveScore;
                end
%                 fprintf('iteration:%d, objectivescore:%f\n', iter, newObjectiveScore);
                diff = oldObjectiveScore - newObjectiveScore;
            end
            fprintf('iteration used: %d\n', iter);
            foldObjectiveScores(fold) = newObjectiveScore;
            
            %calculate validationScore
            [projB, ~] = SumOfMatricize(B, 2*(targetDomain - 1)+1);
            result = U{targetDomain}*projB*V{targetDomain}';
            
            [~, maxIndex] = max(result, [], 2);
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
        
        TotalTime = toc(TotalTimer);
        time = round(clock);
        fprintf('Time: %d/%d/%d,%d:%d:%d\n', time(1), time(2), time(3), time(4), time(5), time(6));
        fprintf('Initial try: %d, ObjectiveScore:%f, Accuracy:%f%%\n', t, avgObjectiveScore, accuracy*100);
        resultCellArray{t}{1} = avgObjectiveScore;
        resultCellArray{t}{2} = accuracy*100;
        resultCellArray{t}{3} = TotalTime;
    end
    for numResult = 1:randomTryTime
        fprintf(resultFile, '%f,%f,%f,%f\n', lambda, resultCellArray{numResult}{1}, resultCellArray{numResult}{2}, resultCellArray{numResult}{3});
    end
end
showExperimentInfo(exp_title, datasetId, prefix, numSampleInstance, numSampleFeature, numInstanceCluster, numFeatureCluster, sigma);
fprintf('done\n');
fclose(resultFile);
% matlabpool close;