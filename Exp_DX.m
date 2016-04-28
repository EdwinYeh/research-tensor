% Please assign datasetId in the commend line
SetParameter;
randomTryTime = 2;
sigmaList = 0.25:0.25:1;
numInstanceClusterList = [5, 15];
numFeatureClusterList = [5, 15];
cpRankList = [15];

lambdaMaxOrder = 4;
gamaMaxOrder = 4;
deltaMaxOrder = 4;

lambdaStart = 10^-12;
gamaStart = 10^-12;
deltaStart = 10^-12;

lambdaScale = 1000;
gamaScale = 1000;
deltaScale = 1000;

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
                            delta = 0;
                            main_DX;
                        end
                    end
                end
            end
        end
    end
end
fclose(resultFile);