% Please assign datasetId in the commend line
SetParameter;
sigmaList = [0.1];
numInstanceClusterList = [5];
numFeatureClusterList = [5];
cpRankList = [5];
lambdaMaxOrder = 10;
gamaMaxOrder = 10;
deltaMaxOrder = 10;
lambdaStart = 10^-8;
gamaStart = 10^-8;
deltaStart = 10^-12;
lambdaScale = 10;
gamaScale = 10;
deltaScale = 10;

expTitle = 'DX';
directoryName = sprintf('../exp_result/%s/%d/', expTitle, datasetId);
mkdir(directoryName);
resultFile = fopen(sprintf('../exp_result/%s.csv', expTitle), 'a');
fprintf(resultFile, 'lambda, gama, objectiveScore, accuracy, convergeTime\n');

for tuneSigma = 1: length(sigmaList)
    sigma = sigmaList(tuneSigma);
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
                                %                     try
                                main_DX;
                                %                     catch exception
                                %                         fprintf('%g,%g,%g,%d,%d,%d\n', lambda, gama, sigma, numInstanceCluster, numFeatureCluster, cpRank);
                                %                         disp(exception.message);
                                %                         continue;
                                %                     end
                            end
                        end
                    end
                end
            end
        end
    end
end
fclose(resultFile);