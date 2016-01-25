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
                    fprintf('iter:%d\n', iter);
                    fprintf('domain:%d\n', i);
                    [projB, threeMatrixB] = SumOfMatricize(B, 2*(i - 1)+1);
                    disp('projB');
                    disp(norm(projB));
                    %bestCPR = FindBestRank(threeMatrixB, 50)
%                     disp('cp start');
                    bestCPR = 20;
%                     cpTimer = tic;
%                     CP = cp_apr(tensor(threeMatrixB), bestCPR, 'printitn', 0, 'alg', 'mu');%parafac_als(tensor(threeMatrixB), bestCPR);
                    CP = cp_als(tensor(threeMatrixB), bestCPR, 'printitn', 0);
%                     disp('cp_end');
%                     cpUsedTime = toc(cpTimer);
%                     disp(cpUsedTime);
                    A = CP.U{1};
                    E = CP.U{2};
                    U3 = CP.U{3};
                    
                    fi = cell(1, length(CP.U{3}));
%                     disp('Update U, V');
%                     updateUVTimer = tic;
                    V{i} = YMatrix{i}'*U{i}*projB/(projB'*U{i}'*U{i}*projB);
                    
%                     %col normalize
%                     [r, ~] = size(V{i});
%                     for tmpI = 1:r
%                         bot = sum(abs(V{i}(tmpI,:)));
%                         if bot == 0
%                             bot = 1;
%                         end
%                         V{i}(tmpI,:) = V{i}(tmpI,:)/bot;
%                     end
                    disp('V');
                    disp(norm(V{i}));
                    if all(all(isnan(V{i})))~=0
                        fprintf('V{%d} has nan\n', i);
                        break;
                    end
                    if all(all(~isfinite(V{i})))~=0
                        fprintf('V{%d} has infinity\n', i);
                        break;
                    end
                    if ~isreal(V{i})
                        fprintf('V{%d} has complex number\n', i);
                        break;
                    end
%                     V{i}(isnan(V{i})) = 0;
%                     V{i}(~isfinite(V{i})) = 0;
                    
                    %update U
                    tmpA = -YMatrix{i}*V{i}*projB';
                    tmpB = projB*V{i}'*V{i}*projB';
                    tmpC = Lu{i};
                    
                    vectorA = reshape(tmpA, [], 1);
                    tmpM = constructM(tmpB, tmpC);
                    vectorU = tmpM\-vectorA;
                    
                    U{i} = reshape(vectorU, [], numInstanceCluster);                  
                    
%                     [r, ~] = size(U{i});
%                     for tmpI = 1:r
%                         bot = sum(abs(U{i}(tmpI,:)));
%                         if bot == 0
%                             bot = 1;
%                         end
%                         U{i}(tmpI,:) = U{i}(tmpI,:)/bot;
%                     end
                    
                    disp('U');
                    disp(norm(U{i}));
                    
                    if all(all(isnan(U{i})))~=0
                        fprintf('U{%d} has nan\n', i);
                        break;
                    end
                    if all(all(~isfinite(U{i})))~=0
                        fprintf('U{%d} has infinity\n', i);
                        break;
                    end
                    if ~isreal(U{i})
                        fprintf('U{%d} has complex number\n', i);
                        break;
                    end
                    
%                     U{i}(isnan(U{i})) = 0;
%                     U{i}(~isfinite(U{i})) = 0;
                    
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
%                         disp('Update A, E');
                        [rA, cA] = size(A);
                        onesA = ones(rA, cA);
                        inverse1 = U{i}'*U{i};
                        inverse2 = sumFi*E'*V{i}'*V{i}*E*sumFi';
                        center = U{i}'*YMatrix{i}*V{i}*E*sumFi';
                        tmp = inverse1\center;
                        A = tmp/inverse2;
                        disp(norm(A));
                        if all(all(isnan(A)))~=0
                            fprintf('A has nan\n');
                            break;
                        end
                        if all(all(~isfinite(A)))~=0
                            fprintf('A has infinity\n');
                            break;
                        end
                        if ~isreal(A)
                            fprintf('A has complex number\n');
                            break;
                        end
%                         A(isnan(A)) = 0;
%                         A(~isfinite(A)) = 0;
                        [rE ,cE] = size(E);
                        onesE = ones(rE, cE);
                        inverse1 = sumFi'*A'*U{i}'*U{i}*A*sumFi;
                        inverse2 = V{i}'*V{i};
                        center = sumFi'*A'*U{i}'*YMatrix{i}*V{i};
                        tmp = inverse1\center;
                        E = (tmp/inverse2)';
                        disp(norm(E));
                        if all(all(isnan(E)))~=0
                            fprintf('E has nan\n');
                            break;
                        end
                        if all(all(~isfinite(E)))~=0
                            fprintf('E has infinity\n');
                            break;
                        end
                        if ~isreal(E)
                            fprintf('E has complex number\n');
                            break;
                        end
%                         E(isnan(E)) = 0;
%                         E(~isfinite(E)) = 0;
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
                fprintf('iteration:%d, objectivescore:%f\n', iter, newObjectiveScore);
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
showExperimentInfo(datasetId, prefix, numSampleInstance, numSampleFeature, numInstanceCluster, numFeatureCluster, sigma);
fprintf('done\n');
fclose(resultFile);
% matlabpool close;