SetParameter;
resultDirectory = sprintf('../exp_result/newmodel/DY_cross_3way/%d/', datasetId);
mkdir(resultDirectory);
expTitle = sprintf('DY_cross_3way_%d', datasetId);
sampleSizeLevel = '1000_100';
resultFile = fopen(sprintf('%s%s.csv', resultDirectory, expTitle), 'a');
fprintf(resultFile, 'cpRank,instanceCluster,beta,gama,lambda,objectiveScore,accuracy1,accuracy2,trainingTime\n');

betaStart = 0;
betaScale = 1000;
betaMaxOrder = 0;

gamaStart = 10^-12;
gamaScale = 10^3;
gamaMaxOrder = 4;

lambdaStart = 10^-6;
lambdaScale = 10;
lambdaMaxOrder = 6;

sigmaList = [0.1, 0.5, 1];
cpRankList = [10,20,50];
instanceClusterList = [10,20,50];

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
                        
                        S = cell(length(X),1);
                        for domID = 1:length(X)
                            S{domID}=ones(numTrainData(domID), numClass(domID));
                            input.Sxw{domID} = Su{domID};
                            input.Dxw{domID} = Du{domID};
                        end
                        input.X = XTrain;
                        input.Y = YTrain;
                        input.S= S;
                        
                        hyperparam.beta = 0;
                        hyperparam.gamma = gama;
                        hyperparam.lambda = lambda;
                        hyperparam.cpRank = cpRank;
                        hyperparam.clusterNum = numInstanceCluster;
                        save environmentForNewModel;
                        trainingTimer = tic;
                        output=solver(input,hyperparam);
                        trainingTime = toc(trainingTimer);
                        
                        trainAccuracy = zeros(1, numDom);
                        testAccuracy = zeros(1, numDom);
                        for domID = 1:numDom
                            trainAccuracy(domID) = comparePredictResult(YTrain{domID}, output.reconstrucY{domID});
                            testLabel = predict(output, XTest{domID}, domID);
                            testAccuracy(domID) = comparePredictResult(YTest{domID}, testLabel);
                            fprintf('domain: %d, trainAccuracy: %g, testAccuracy: %g\n', domID, trainAccuracy(domID), testAccuracy(domID));
                        end
                        fprintf(resultFile, '%g,%g,%g,%g,%g,%g,%g,%g,%g\n', cpRank, numInstanceCluster, 0, gama, lambda, output.objective, testAccuracy(1), testAccuracy(2), trainingTime);
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
