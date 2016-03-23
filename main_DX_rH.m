% if parpool('size') > 0
%     parpool close;
% end
% parpool('open', 'local', 4);

% disp('Start training');
if isMakePrediction
    resultFile = fopen(sprintf('../exp_result/%s_%d.csv', expTitle, datasetId), 'a');
    fprintf(resultFile, 'delta, lambda, gama, sigma, instanceCluster, featureCluster, cpRank\n');
end

for tuneGama = 0:6
    gama = 0.00000001 * 10 ^ tuneGama;
    for tuneLambda = 0:6
        lambda = 0.00000001 * 10 ^ tuneLambda;
        for tuneDelta = 0:4
            delta = 0.000001 * 100 ^ tuneDelta;
            time = round(clock);
            fprintf('Time: %d/%d/%d,%d:%d:%d\n', time(1), time(2), time(3), time(4), time(5), time(6));
            fprintf('Use (%g_%g_%g_%g_%d_%d_%d)\n', delta, lambda, gama, sigma, numInstanceCluster, numFeatureCluster, cpRank);
            bestRandomInitialObjectiveScore = Inf;
            for t = 1: randomTryTime
                objectiveScore = 0;
                %Iterative update
                U = initU(t, :);
                V = initV(t, :);
                CP1 = rand(numInstanceCluster, cpRank);
                CP2 = rand(numFeatureCluster, cpRank);
                CP3 = rand(numInstanceCluster, cpRank);
                CP4 = rand(numFeatureCluster, cpRank);
                newObjectiveScore = Inf;
                iter = 0;
                diff = Inf;
                convergeTimer = tic;
                while (diff >= 0.0001  && iter < maxIter)
                    iter = iter + 1;
                    oldObjectiveScore = newObjectiveScore;
                    % Display information of per iteration
%                     fprintf('#Iterator:%d, objectiveScore:%f\n', iter, newObjectiveScore);
                    %--------------------------------------------------
                    newObjectiveScore = 0;
                    for dom = 1:numDom
                        % Compute projection of tensor by domain
                        [A,sumFi,E] = projectTensorToMatrix({CP1,CP2,CP3,CP4}, dom);
                        projB = A*sumFi*E';
%                         fprintf('rank H: %d\n', rank(projB));
%                         fprintf('norm H: %f\n', norm(projB, 'fro'));
                        %---------------------------------------------------------
                        
                        % Update V
                        V{dom} = V{dom}.*sqrt((X{dom}'*U{dom}*projB + gama*Sv{dom}*V{dom})./(V{dom}*projB'*U{dom}'*U{dom}*projB + gama*Dv{dom}*V{dom}));
                        V{dom}(isnan(V{dom})) = 0;
                        V{dom}(~isfinite(V{dom})) = 0;
                        % col normalize
                        [r, ~] = size(V{dom});
                        for tmpI = 1:r
                            bot = sum(abs(V{dom}(tmpI,:)));
                            if bot == 0
                                bot = 1;
                            end
                            V{dom}(tmpI,:) = V{dom}(tmpI,:)/bot;
                        end
                        V{dom}(isnan(V{dom})) = 0;
                        V{dom}(~isfinite(V{dom})) = 0;
                        
                        %update U
                        U{dom} = U{dom}.*sqrt((X{dom}*V{dom}*projB'+lambda*Su{dom}*U{dom})./(U{dom}*projB*V{dom}'*V{dom}*projB'+lambda*Du{dom}*U{dom}));
                        U{dom}(isnan(U{dom})) = 0;
                        U{dom}(~isfinite(U{dom})) = 0;
                        %col normalize
                        [r ,~] = size(U{dom});
                        for tmpI = 1:r
                            bot = sum(abs(U{dom}(tmpI,:)));
                            if bot == 0
                                bot = 1;
                            end
                            U{dom}(tmpI,:) = U{dom}(tmpI,:)/bot;
                        end
                        U{dom}(isnan(U{dom})) = 0;
                        U{dom}(~isfinite(U{dom})) = 0;
                        
                        %update AE
                        %disp(sprintf('\t\tupdate A...'));
                        [rA, cA] = size(A);
                        onesA = ones(rA, cA);
                        A = A.*sqrt((U{dom}'*X{dom}*V{dom}*E*sumFi + alpha*(onesA))./(delta*(A*sumFi*(E'*E)*sumFi'/norm(A*sumFi*E','fro'))+U{dom}'*U{dom}*A*sumFi*E'*V{dom}'*V{dom}*E*sumFi));
                        A(isnan(A)) = 0;
                        A(~isfinite(A)) = 0;
                        %A = (spdiags (sum(abs(A),1)', 0, cA, cA)\A')';
                        A(isnan(A)) = 0;
                        A(~isfinite(A)) = 0;
                        
                        %disp(sprintf('\t\tupdate E...'));
                        [rE ,cE] = size(E);
                        onesE = ones(rE, cE);
                        E = E.*sqrt((V{dom}'*X{dom}'*U{dom}*A*sumFi + beta*(onesE))./(delta*(E*sumFi'*(A'*A)*sumFi/norm(A*sumFi*E','fro'))+V{dom}'*V{dom}*E*sumFi*A'*U{dom}'*U{dom}*A*sumFi));
                        E(isnan(E)) = 0;
                        E(~isfinite(E)) = 0;
                        %E = (spdiags (sum(abs(E),1)', 0, cE, cE)\E')';
                        E(isnan(E)) = 0;
                        E(~isfinite(E)) = 0;
                        
                        if dom == sourceDomain
                            CP1 = A;
                            CP2 = E;
                        else
                            CP3 = A;
                            CP4 = E;
                        end
                    end
                    %disp(sprintf('\tCalculate this iterator error'));
                    for dom = 1:numDom
                        %for i = 1:numDom
                        [A,sumFi,E] = projectTensorToMatrix({CP1,CP2,CP3,CP4}, dom);
                        projB = A*sumFi*E';
                        result = U{dom}*projB*V{dom}';
                        normEmp = norm((X{dom} - result), 'fro')*norm((X{dom} - result), 'fro');
                        smoothU = lambda*trace(U{dom}'*Lu{dom}*U{dom});
                        smoothV = gama*trace(V{dom}'*Lv{dom}*V{dom});
                        normH = delta*norm(projB, 'fro');
                        objectiveScore = normEmp + smoothU + smoothV + normH;
                        newObjectiveScore = newObjectiveScore + objectiveScore;
                        %disp(sprintf('\t\tdomain #%d => empTerm:%f, smoothU:%f, smoothV:%f ==> objective score:%f', i, normEmp, smoothU, smoothV, objectiveScore));
                    end
                    %disp(sprintf('\tEmperical Error:%f', newObjectiveScore));
                    %fprintf('iter:%d, error = %f\n', iter, newObjectiveScore);
                    diff = oldObjectiveScore - newObjectiveScore;
                end
                convergeTime = toc(convergeTimer);
                if newObjectiveScore < bestRandomInitialObjectiveScore
                    bestRandomInitialObjectiveScore = newObjectiveScore;
                    bestU = U;
                    bestConvergeTime = convergeTime;
                end
            end
            save(sprintf('%sU_%g_%g_%g_%g_%d_%d_%d.mat', directoryName, delta, lambda, gama, sigma, numInstanceCluster, numFeatureCluster, cpRank), 'bestU');
            if isMakePrediction
                targetTestingDataIndex = 1:CVFoldSize;
                numCorrectPredict = 0;
                for cvFold = 1: numCVFold
                    targetTrainingDataIndex = setdiff(1:numInstance(targetDomain),targetTestingDataIndex);
                    trainingData = [bestU{sourceDomain}; bestU{targetDomain}(targetTrainingDataIndex,:)];
                    trainingLabel = [sampledLabel{sourceDomain}; sampledLabel{targetDomain}(targetTrainingDataIndex, :)];
                    svmModel = fitcsvm(trainingData, trainingLabel, 'KernelFunction', 'rbf');
                    predictLabel = predict(svmModel, bestU{targetDomain}(targetTestingDataIndex,:));
                    for dataIndex = 1: CVFoldSize
                        if sampledLabel{targetDomain}(targetTestingDataIndex(dataIndex)) == predictLabel(dataIndex)
                            numCorrectPredict = numCorrectPredict + 1;
                        end
                    end
                    targetTestingDataIndex = targetTestingDataIndex + CVFoldSize;
                end
                accuracy = numCorrectPredict/ (CVFoldSize*numCVFold);
                fprintf('Delta:%g, Lambda:%g, Gama:%g, Sigma:%g, InstanceCluster:%d, FeatureCluster:%d, cpRank:%d\n', delta, lambda, gama, sigma, numInstanceCluster, numFeatureCluster, cpRank);
                fprintf('ObjectiveScore:%g, Accuracy:%g%%\n', bestRandomInitialObjectiveScore, accuracy);
                fprintf(resultFile, '%f,%g,%g,%g,%g,%d,%d,%d,%f\n', accuracy, delta, lambda, gama, sigma, numInstanceCluster, numFeatureCluster, cpRank, bestConvergeTime);
            end
        end
    end
end
if isMakePrediction
    fclose(resultFile);
end
%showExperimentInfo(exp_title, datasetId, prefix, numSourceInstanceList, numTargetInstanceList, numSourceFeatureList, numTargetFeatureList, numSampleInstance, numSampleFeature);
fprintf('done\n');
% parpool close;

