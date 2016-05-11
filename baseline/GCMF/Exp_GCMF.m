% Please assign datasetId in the commend line
SetParameter;
randomTryTime = 1;
maxIter = 300;
sigmaList = [0.0001, 0.001, 0.1];
sigma2List = [0.0001, 0.001, 0.1];
numFeatureClusterList = [10];

lambdaStart = 10^-3;
gamaStart = 10^-3;

lambdaScale = 1000;
gamaScale = 1000;

lambdaMaxOrder = 2;
gamaMaxOrder = 2;

expTitle = sprintf('GCMF%d', datasetId);
resultDirectory = sprintf('../../../exp_result/GCMF/%d/', datasetId);
mkdir(resultDirectory);
resultFile = fopen(sprintf('%s%s_validate.csv', resultDirectory, expTitle), 'a');
fprintf(resultFile, 'numInstanceCluster, numFeatureCluster, sigma, sigma2, lambda, gama, objectiveScore, accuracy, convergeTime\n');

isTestPhase = false;
for tuneSigma = 1: length(sigmaList)
    sigma = sigmaList(tuneSigma);
    for tuneSigma2 = 1:length(sigma2List)
        sigma2 = sigma2List(tuneSigma2);
        PrepareGCMFExperiment;
        isLoad = true;
        for tuneNumFeatureCluster = 1: length(numFeatureClusterList)
            numFeatureCluster = numFeatureClusterList(tuneNumFeatureCluster);
            numInstanceCluster = 2;
            for lambdaOrder = 0: lambdaMaxOrder
                lambda = lambdaStart * lambdaScale^lambdaOrder;
                for gamaOrder = 0: gamaMaxOrder
                    gama = gamaStart * gamaScale^gamaOrder;
                    main_GCMF_beta;
                end
            end
        end
    end
end
fclose(resultFile);
isTestPhase = true;
randomTryTime = 10;
numCVFold = 1;
maxIter = 1000;
resultFile = fopen(sprintf('%s%s_test.csv', resultDirectory, expTitle), 'a');
fprintf(resultFile, 'numInstanceCluster, numFeatureCluster, sigma, sigma2, lambda, gama, objectiveScore, accuracy, convergeTime\n');
load(sprintf('%sBestParameter_%s.mat', resultDirectory, expTitle));
PrepareGCMFExperiment;
main_GCMF_beta;
fclose(resultFile);