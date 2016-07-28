SetParameter;
resultDirectory = sprintf('../exp_result/newmodel/%d/', datasetId);
mkdir(resultDirectory);
expTitle = sprintf('newmodel_%d', datasetId);
sampleSizeLevel = '1000_100';
resultFile = fopen(sprintf('%s%s.csv', resultDirectory, expTitle), 'a');
fprintf(resultFile, 'sigma,cpRank,instanceCluster,beta,gama,lambda,objectiveScore,accuracy1,accuracy2,trainingTime\n');

betaStart = 0;
betaScale = 1000;
betaMaxOrder = 0;

gamaStart = 10^-12;
gamaScale = 10^3;
gamaMaxOrder = 0;

lambdaStart = 10^-7;
lambdaScale = 10^2;
lambdaMaxOrder = 0;

sigmaList = [0.005];
cpRankList = [10];
instanceClusterList = [10];

for tuneSigma = 1:length(sigmaList)
    sigma = sigmaList(tuneSigma);
    PrepareExperimentNew;
    for tuneCPRank = 1: length(cpRankList)
        cpRank = cpRankList(tuneCPRank);
        for tuneInstanceCluster = 1: length(instanceClusterList)
            numInstanceCluster = instanceClusterList(tuneInstanceCluster);
            if numInstanceCluster <= cpRank
                for lambdaOrder = 0: lambdaMaxOrder
                    lambda = lambdaStart * lambdaScale ^ lambdaOrder;
                    for gamaOrder = 0: gamaMaxOrder
                        gama = gamaStart * gamaScale ^ gamaOrder;
                        
                        CVFoldSize = numTrainData(1)/CVFoldNum;
                        validationIndex = cell(1,2);
                        trainIndex = cell(1,2);
                        for domID = 1:numDom
                            validationIndex{domID} = 1:CVFoldSize;
                        end
                        avgValidationAccuracy = zeros(numDom, 1);
                        trainingTime = 0;
                        
                        for cvFoldDom1 = 1:CVFoldNum
                            for cvFoldDom2 = 1:CVFoldNum                               
                                % S: selection matrix for validation set
                                for domID = 1:length(X)
                                    trainIndex = setdiff(1:numTrainData(domID), validationIndex{domID});
                                    fprintf('validation index domain%d: %d~%d\n', domID, min(validationIndex{domID}), max(validationIndex{domID}));
                                    input.S{domID}=ones(length(trainIndex), numClass(domID));
                                    input.X{domID} = XTrain{domID}(trainIndex, :);
                                    input.Y{domID} = YTrain{domID}(trainIndex, :);
                                    input.Sxw{domID} = Su{domID}(trainIndex, trainIndex);
                                    input.Dxw{domID} = Du{domID}(trainIndex, trainIndex);
                                end;
                                
                                hyperparam.beta = 0;
                                hyperparam.gamma = gama;
                                hyperparam.lambda = lambda;
                                hyperparam.cpRank = cpRank;
                                hyperparam.clusterNum = numInstanceCluster;
                                
                                trainingTimer = tic;
                                output=solver(input,hyperparam);
                                trainingTime = trainingTime + toc(trainingTimer);
                                
                                validationAccuracy = zeros(1, numDom);
                                for domID = 1:numDom
                                    %                                     validationAccuracy(domID) = comparePredictResult(YTrain{domID}(validationIndex,:), output.reconstrucY{domID}(validationIndex,:));
                                    validationLabel = predict(output, XTrain{domID}(validationIndex{domID}, :), domID);
                                    validationAccuracy(domID) = comparePredictResult(YTrain{domID}(validationIndex{domID}, :), validationLabel);
                                    avgValidationAccuracy(domID) = avgValidationAccuracy(domID) + validationAccuracy(domID);
                                    %                                 fprintf('domain: %d, validationAccuracy: %g\n', domID, validationAccuracy(domID));
                                end
                                disp(validationAccuracy);
                                validationIndex{2} = validationIndex{2} + CVFoldSize;
                            end
                            validationIndex{2} = 1:CVFoldSize;
                            validationIndex{1} = validationIndex{1} + CVFoldSize;
                        end
                        trainingTime = trainingTime/(CVFoldNum*CVFoldNum);
                        avgValidationAccuracy = avgValidationAccuracy/(CVFoldNum*CVFoldNum);
                        fprintf(resultFile, '%g,%g,%g,%g,%g,%g,%g,%g,%g,%g\n', sigma, cpRank, numInstanceCluster, 0, gama, lambda, output.objective, avgValidationAccuracy(1), avgValidationAccuracy(2), trainingTime);
                        fprintf('%g,%g,%g,%g,%g,%g,%g,%g,%g,%g\n', sigma, cpRank, numInstanceCluster, 0, gama, lambda, output.objective, avgValidationAccuracy(1), avgValidationAccuracy(2), trainingTime);
                    end
                end
            end
        end
    end
end
fclose(resultFile);
%     disp('Start testing');
%     isTestPhase = true;
%     randomTryTime = 1;
%     resultFile = fopen(sprintf('%s%s_test.csv', resultDirectory, expTitle), 'a');
%     fprintf(resultFile, 'cpRank,instanceCluster,featureCluster,sigma,lambda,delta,objectiveScore,accuracy,trainingTime\n');
%     load(sprintf('%sBestParameter_%s.mat', resultDirectory, expTitle));
%     PrepareExperiment;
%     main_DY_cross_domain_3way;
%     fclose(resultFile);
