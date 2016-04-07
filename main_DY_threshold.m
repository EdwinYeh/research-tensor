disp('Start training');

fprintf(resultFile,'accuracy,sigma,lambda,delta,trainingTime\n');

lambdaScale = 10;
deltaScale = 10;
for lambdaOrder = 0:6
    lambda = 10^(-6)* lambdaScale^lambdaOrder;
    for deltaOrder = 0:8
        delta = 10^(-8)* deltaScale^deltaOrder;
        time = round(clock);
        fprintf('Time: %d/%d/%d,%d:%d:%d\n', time(1), time(2), time(3), time(4), time(5), time(6));
        fprintf('Use (sigma,lambda,delta)=(%g,%g,%g)\n', sigma, lambda, delta);
        % Random initila several times
        % and take the result with minimal objective score
        bestObjectiveScore = Inf;
        bestTrainAndPredictTime = 0;
        bestAccuracy = 0;
        for t = 1: randomTryTime
            numCorrectPredict = 0;
            validateIndex = 1: CVFoldSize;
            foldObjectiveScores = zeros(1,numCVFold);
            trainAndPredictTimer = tic;
            for fold = 1:numCVFold
                YMatrix = TrueYMatrix;
                YMatrix{targetDomain}(validateIndex, :) = zeros(CVFoldSize, numClass(1));
                W = ones(numSampleInstance(targetDomain), numClass(1));
                W(validateIndex, :) = 0;
                U = initU(t, :);
                V = initV(t, :);
                CP1 = rand(numInstanceCluster, cpRank);
                CP2 = rand(numFeatureCluster, cpRank);
                CP3 = rand(numInstanceCluster, cpRank);
                CP4 = rand(numFeatureCluster, cpRank);
                iter = 0;
                diff = Inf;
                newObjectiveScore = 0;
                oldObjectiveScore = Inf;
                while (diff >= 0.0001  && iter < maxIter)
%                     fprintf('Random:%d, Fold:%d, Iter:%d, Objective:%g\n', t, fold, iter, newObjectiveScore);
                    iter = iter + 1;
                    newObjectiveScore = 0;
                    for domId = 1:numDom
                        [A,sumFi,E] = projectTensorToMatrix({CP1,CP2,CP3,CP4}, dom);
                        projB = A*sumFi*E';
                        
                        if domId == targetDomain
                            V{domId} = V{domId}.*sqrt((YMatrix{domId}'*U{domId}*projB)./((V{domId}*projB'*U{domId}'.*W')*U{domId}*projB));
                        else
                            V{domId} = V{domId}.*sqrt((YMatrix{domId}'*U{domId}*projB)./((V{domId}*projB'*U{domId}')*U{domId}*projB));
                        end
                        V{domId}(isnan(V{domId})) = 0;
                        V{domId}(~isfinite(V{domId})) = 0;
                        
                        %col normalize
                        [r, ~] = size(V{domId});
                        for tmpI = 1:r
                            bot = sum(abs(V{domId}(tmpI,:)));
                            if bot == 0
                                bot = 1;
                            end
                            V{domId}(tmpI,:) = V{domId}(tmpI,:)/bot;
                        end
                        V{domId}(isnan(V{domId})) = 0;
                        V{domId}(~isfinite(V{domId})) = 0;
                        
                        %update U
                        if domId == targetDomain
                            U{domId} = U{domId}.*sqrt((YMatrix{domId}*V{domId}*projB' + lambda*Su{domId}*U{domId})./((U{domId}*projB*V{domId}'.*W)*V{domId}*projB' + lambda*Du{domId}*U{domId}));
                        else
                            U{domId} = U{domId}.*sqrt((YMatrix{domId}*V{domId}*projB' + lambda*Su{domId}*U{domId})./(U{domId}*projB*V{domId}'*V{domId}*projB' + lambda*Du{domId}*U{domId}));
                        end
                        U{domId}(isnan(U{domId})) = 0;
                        U{domId}(~isfinite(U{domId})) = 0;
                        
                        %col normalize
                        [r, ~] = size(U{domId});
                        for tmpI = 1:r
                            bot = sum(abs(U{domId}(tmpI,:)));
                            if bot == 0
                                bot = 1;
                            end
                            U{domId}(tmpI,:) = U{domId}(tmpI,:)/bot;
                        end
                        U{domId}(isnan(U{domId})) = 0;
                        U{domId}(~isfinite(U{domId})) = 0;
                        
                        % Update AE
                        
                        % Indicator matrix where entries >= delta = 1,
                        % otherwise = 0
                        
                        [rA, cA] = size(A);
                        onesA = ones(rA, cA);
                        A = A.*sqrt((U{domId}'*YMatrix{domId}*V{domId}*E*sumFi+alpha*(onesA))./(U{domId}'*U{domId}*A*sumFi*E'*V{domId}'*V{domId}*E*sumFi));
                        wthresh(A, 's', delta);
                        A(isnan(A)) = 0;
                        A(~isfinite(A)) = 0;
                        
                        [rE ,cE] = size(E);
                        onesE = ones(rE, cE);
                        E = E.*sqrt((V{domId}'*YMatrix{domId}'*U{domId}*A*sumFi + beta*(onesE))./(V{domId}'*V{domId}*E*sumFi*A'*U{domId}'*U{domId}*A*sumFi));
                        wthresh(E, 's', delta);
                        E(isnan(E)) = 0;
                        E(~isfinite(E)) = 0;
                        if domId == sourceDomain
                            CP1 = A;
                            CP2 = E;
                        else
                            CP3 = A;
                            CP4 = E;
                        end
                    end
                    for domId = 1:numDom
                        [A,sumFi,E] = projectTensorToMatrix({CP1,CP2,CP3,CP4}, dom);
                        projB = A*sumFi*E';
                        result = U{domId}*projB*V{domId}';
                        if domId == targetDomain
                            normEmp = norm((YMatrix{domId} - result).*W, 'fro')*norm((YMatrix{domId} - result).*W, 'fro');
                        else
                            normEmp = norm((YMatrix{domId} - result), 'fro')*norm((YMatrix{domId} - result), 'fro');
                        end
                        smoothU = lambda*trace(U{domId}'*Lu{domId}*U{domId});
                        normH = delta*norm(projB, 1);
                        objectiveScore = normEmp + smoothU + normH;
                        newObjectiveScore = newObjectiveScore + objectiveScore;
                    end
                    diff = oldObjectiveScore - newObjectiveScore;
                    oldObjectiveScore = newObjectiveScore;
                end
                foldObjectiveScores(fold) = newObjectiveScore;
                
                %calculate validationScore
                [A,sumFi,E] = projectTensorToMatrix({CP1,CP2,CP3,CP4}, dom);
                projB = A*sumFi*E';
                result = U{targetDomain}*projB*V{targetDomain}';
                
                [~, maxIndex] = max(result, [], 2);
                predictResult = maxIndex;
                for instanceId = 1: CVFoldSize
                    if(predictResult(validateIndex(instanceId)) == Label{targetDomain}(validateIndex(instanceId)))
                        numCorrectPredict = numCorrectPredict + 1;
                    end
                end
                validateIndex = validateIndex + CVFoldSize;
            end
            trainAndPredictTime = toc(trainAndPredictTimer);
            
            accuracy = numCorrectPredict/ numSampleInstance(targetDomain);
            avgObjectiveScore = sum(foldObjectiveScores)/ numCVFold;
            
            if avgObjectiveScore < bestObjectiveScore
                bestObjectiveScore = avgObjectiveScore;
                bestAccuracy = accuracy;
                bestTrainAndPredictTime = trainAndPredictTime;
            end         
        end
        fprintf(resultFile, '%f,%g,%g,%g,%f\n', bestAccuracy, sigma, lambda, delta, bestTrainAndPredictTime);
        % matlabpool close;
    end
end
fprintf('done\n\n');