% Please assign datasetId in the commend line
SetParameter;
randomTryTime = 1;
sigmaList = 0.01:0.03:0.1;
sigma2List = 0.01:0.03:0.1;
numInstanceClusterList = [10];
numFeatureClusterList = [10];
cpRankList = [10];

lambdaStart = 10^-3;
gamaStart = 10^-3;
deltaStart = 10^-15;

lambdaScale = 1000;
gamaScale = 1000;
deltaScale = 1000;

lambdaMaxOrder = 2;
gamaMaxOrder = 2;
deltaMaxOrder = 3;

expTitle = sprintf('DX%d', datasetId);
resultDirectory = sprintf('../exp_result/DX_row/%d/', datasetId);
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
% isTestPhase = true;
% randomTryTime = 10;
% resultFile = fopen(sprintf('%s%s_test.csv', resultDirectory, expTitle), 'a');
% fprintf(resultFile, 'cpRank, numInstanceCluster, numFeatureCluster, sigma, sigma2, lambda, gama, delta, objectiveScore, accuracy, convergeTime\n');
% load(sprintf('%sBestParameter_%s.mat', resultDirectory, expTitle));
% PrepareExperiment;
% main_DX_row;
% fclose(resultFile);