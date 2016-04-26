SetParameter;
isTestPhase = true;
exp_title = sprintf('DY_onenorm_%d', datasetId);
resultFile = fopen(sprintf('../exp_result/%s.csv', exp_title), 'a');
fprintf(resultFile, 'cpRank,instanceCluster,featureCluster,sigma,sigma2,lambda,delta,objectiveScore,accuracy,trainingTime\n');
lambdaStart = 10^-9;
lambdaMaxOrder = 0;
lambdaScale = 1000;
deltaStart = 10^-9;
deltaMaxOrder = 0;
deltaScale = 1000;
randomTryTime = 1;
sigmaList = 0.02:0.02:0.02;
cpRankList = 10:10:10;
instanceClusterList = 10:10:10;
featureClusterList = 10:10:10;
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
                fclose(resultFile);
            end
        end
    end
end