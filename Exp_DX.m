% Please assign datasetId in the commend line
SetParameter;
randomTryTime = 2;
sigmaList = 0.25:0.25:1;
numInstanceClusterList = [5, 10, 15, 30];
numFeatureClusterList = [5, 10, 15, 30];
cpRankList = [5, 10, 15, 30];
lambdaMaxOrder = 5;
gamaMaxOrder = 5;
deltaMaxOrder = 3;

lambdaStart = 10^-8;
gamaStart = 10^-8;
deltaStart = 10^-8;

lambdaScale = 100;
gamaScale = 100;
deltaScale = 100;

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
                                if numInstanceCluster <= cpRank && numFeatureCluster <= cpRank
                                    main_DX;
                                end
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