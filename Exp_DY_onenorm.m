SetParameter;
isTestPhase = true;
exp_title = sprintf('DY_onenorm_%d', datasetId);
resultFile = fopen(sprintf('../exp_result/%s.csv', exp_title), 'a');
fprintf(resultFile, 'cpRank,instanceCluster,featureCluster,sigma,sigma2,lambda,delta,objectiveScore,accuracy,trainingTime\n');
lambdaStart = 10^-12;
lambdaMaxOrder = 8;
lambdaScale = 100;
deltaStart = 10^-12;
deltaMaxOrder = 8;
deltaScale = 100;
randomTryTime = 5;
sigmaList = 0.25:0.25:1;
cpRankList = [10];
instanceClusterList = [5, 10];
featureClusterList = [5, 10];
sigma2 = -1;
for tuneSigma = 1:length(sigmaList)
    sigma = sigmaList(tuneSigma);
    PrepareExperiment;
    for tuneCPRank = 1: length(cpRankList)
        cpRank = cpRankList(tuneCPRank);
        for tuneInstanceCluster = 1: length(instanceClusterList)
            numInstanceCluster = instanceClusterList(tuneInstanceCluster);
            for tuneFeatureCluster = 1: length(featureClusterList)
                numFeatureCluster = featureClusterList(tuneFeatureCluster);
                if numInstanceCluster <= cpRank && numFeatureCluster <= cpRank
                    for lambdaOrder = 0: lambdaMaxOrder
                        lambda = lambdaStart * lambdaScale ^ lambdaOrder;
                        for deltaOrder = 0: deltaMaxOrder
                            delta = deltaStart * deltaScale ^ deltaOrder;
                            main_DY_onenorm;
                        end
                        delta = 0;
                        main_DY_onenorm;
                    end
                end
            end
        end
    end
end
fclose(resultFile);