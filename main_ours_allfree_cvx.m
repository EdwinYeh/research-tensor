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
                    
%                     %Solve cvx U
%                     cvx_begin
%                         variable tmpU(size(U{i}));
%                         minimize(norm(YMatrix{i}-tmpU*projB*V{i}')+trace(tmpU'*Lu{i}*tmpU));
%                     cvx_end
%                     Assign cvx result
                    U{i} = tmpU;
                    disp('Update U');
                    tmpA = -YMatrix{i}*V{i}*projB';
                    tmpB = projB*V{i}'*V{i}*projB';
                    tmpC = Lu{i};
                    
                    vectorA = reshape(tmpA, [], 1);
                    tmpM = constructM(tmpB, tmpC);
                    vectorU = tmpM\-vectorA;
                    
                    U{i} = reshape(vectorU, [], numInstanceCluster);
                    
                    disp(norm(U{i}));
                    
                    disp('Solve cvx V');
                    cvx_begin quiet
                        variable tmpV(size(V{i}));
                        minimize(norm(YMatrix{i}-U{i}*projB*tmpV'));
                    cvx_end
                    % Assign cvx result
                    V{i} = tmpV;
                    disp(norm(V{i}));

                    %Update fi
                    bestCPR = 20;
                    CP = cp_als(tensor(threeMatrixB), bestCPR, 'printitn', 0);                 
                    A = CP.U{1};
                    E = CP.U{2};
                    U3 = CP.U{3};
                    
                    fi = cell(1, length(CP.U{3}));
                    [r, c] = size(U3);
                    nextThreeB = zeros(numInstanceCluster, numFeatureCluster, r);
                    sumFi = zeros(c, c);
                    CPLamda = CP.lambda(:);
                    for idx = 1:r
                        fi{idx} = diag(CPLamda.*U3(idx,:)');
                        sumFi = sumFi + fi{idx};
                    end
                    %Update A, E
                    if isUpdateAE
                        disp('Solve cvx A');
                        cvx_begin quiet
                            variable tmpA(size(A));
                        
                            minimize(norm(YMatrix{i}-U{i}*tmpA*sumFi*E'*V{i}'));
                        cvx_end
                        % Assign cvx result
                        A = tmpA;
                        disp(norm(A));
                        
                        disp('Solve cvx E');
                        cvx_begin quiet
                            variable tmpE(size(E));
                        
                            minimize(norm(YMatrix{i}-U{i}*A*sumFi*tmpE'*V{i}'));
                        cvx_end
                        % Assign cvx result
                        E = tmpE;
                        disp(norm(E));
                        
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
showExperimentInfo(exp_title, datasetId, prefix, numSampleInstance, numSampleFeature, numInstanceCluster, numFeatureCluster, sigma);
fprintf('done\n');
fclose(resultFile);
% matlabpool close;