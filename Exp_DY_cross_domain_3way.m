SetParameter;
resultDirectory = sprintf('../exp_result/DY_cross_3way/%d/', datasetId);
mkdir(resultDirectory);
expTitle = sprintf('DY_cross_3way_%d', datasetId);
resultFile = fopen(sprintf('%s%s_validate.csv', resultDirectory, expTitle), 'a');
fprintf(resultFile, 'cpRank,instanceCluster,featureCluster,sigma,lambda,delta,objectiveScore,accuracy,trainingTime\n');
lambdaStart = 10^-4;
lambdaMaxOrder = 10;%
lambdaScale = 100;
deltaStart = 10^-13;
deltaMaxOrder = 0; %
deltaScale = 1000;
randomTryTime = 3;
sigmaList = 0.015; %-----
cpRankList = [10];
instanceClusterList = [10];
featureClusterList = [10];
isTestPhase = false;
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
                            main_DY_cross_domain_3way;
                        end
                    end
                end
            end
        end
    end
end
fclose(resultFile);
disp('Start testing');
isTestPhase = true;
randomTryTime = 1;
resultFile = fopen(sprintf('%s%s_test.csv', resultDirectory, expTitle), 'a');
fprintf(resultFile, 'cpRank,instanceCluster,featureCluster,sigma,lambda,delta,objectiveScore,accuracy,trainingTime\n');
load(sprintf('%sBestParameter_%s.mat', resultDirectory, expTitle));
PrepareExperiment;
main_DY_cross_domain_3way;
fclose(resultFile);
