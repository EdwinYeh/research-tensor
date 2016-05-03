% Please assign datasetId in the commend line
SetParameter;
randomTryTime = 2;
sigmaList = 0.05:0.05:1;
sigma2List = 0.05:0.05:1;
numInstanceClusterList = [15];
numFeatureClusterList = [15];
cpRankList = [15];

lambdaMaxOrder = 3;
gamaMaxOrder = 3;
deltaMaxOrder = 3;

lambdaStart = 10^-8;
gamaStart = 10^-8;
deltaStart = 10^-8;

lambdaScale = 1000;
gamaScale = 1000;
deltaScale = 1000;

expTitle = sprintf('DX%d', datasetId);
resultDirectory = sprintf('../exp_result/%s/%d/', expTitle, datasetId);
mkdir(resultDirectory);
resultFile = fopen(sprintf('%s%s_validate.csv', resultDirectory, expTitle), 'a');
fprintf(resultFile, 'cpRank, numInstanceCluster, numFeatureCluster, sigma, sigma2, lambda, gama, delta, objectiveScore, accuracy, convergeTime\n');

isTestPhase = false;
for tuneSigma = 1: length(sigmaList)
    sigma = sigmaList(tuneSigma);
    for tuneSigma2 = 1:length(sigma2List)
        sigma2 = sigma2List(tuneSigma2);
        prepareDXExperiment;
        for tuneCPRank = 1: length(cpRankList)
            cpRank = cpRankList(tuneCPRank);
            for tuneNumFeatureCluster = 1: length(numFeatureClusterList)
                numFeatureCluster = numFeatureClusterList(tuneNumFeatureCluster);
                for tuneNumInstanceCluster = 1: length(numInstanceClusterList)
                    numInstanceCluster = numInstanceClusterList(tuneNumInstanceCluster);
                    if numInstanceCluster <= cpRank && numFeatureCluster <= cpRank
                        for lambdaOrder = 0: lambdaMaxOrder
                            lambda = lambdaStart * lambdaScale^lambdaOrder;
                            for gamaOrder = 0: gamaMaxOrder
                                gama = gamaStart * gamaScale^gamaOrder;
                                for deltaOrder = 0: deltaMaxOrder
                                    delta = deltaStart * deltaScale^deltaOrder;
                                    if numInstanceCluster <= cpRank && numFeatureCluster <= cpRank
                                        main_DX_row;
                                    end
                                end
                            end
                        end
                    end
                end
            end
        end
    end
end
fclose(resultFile);
isTestPhase = true;
randomTryTime = 10;
resultFile = fopen(sprintf('%s%s_test.csv', resultDirectory, expTitle), 'a');
fprintf(resultFile, 'cpRank, numInstanceCluster, numFeatureCluster, sigma, sigma2, lambda, gama, delta, objectiveScore, accuracy, convergeTime\n');
load(sprintf('%sBestParameter_%s.mat', resultDirectory, expTitle));
PrepareExperiment;
main_DX_row;
fclose(resultFile);