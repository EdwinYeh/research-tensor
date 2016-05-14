% Please assign datasetId in the commend line
SetParameter;
randomTryTime = 1;
maxIter = 300;
sigmaList = [0.0001, 0.001];
sigma2List = [0.0001, 0.001];
numFeatureClusterList = [10];

lambdaStart = 10^-10;
gamaStart = 10^-10;

lambdaScale = 1000;
gamaScale = 1000;

lambdaMaxOrder = 2;
gamaMaxOrder = 2;

expTitle = sprintf('GCMF%d', datasetId);
resultDirectory = sprintf('../../../exp_result/GCMF/%d/', datasetId);
mkdir(resultDirectory);
resultFile = fopen(sprintf('%s%s_validate.csv', resultDirectory, expTitle), 'w');
fprintf(resultFile, 'numInstanceCluster, numFeatureCluster, sigma, sigma2, lambda, gama, objectiveScore, accuracy, convergeTime\n');

isTestPhase = false;
for tuneSigma = 1: length(sigmaList)
    sigma = sigmaList(tuneSigma);
    for tuneSigma2 = 1:length(sigma2List)
        sigma2 = sigma2List(tuneSigma2);
        PrepareGCMFExperiment;
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
disp('Teting phase');
isTestPhase = true;
randomTryTime = 10;
numCVFold = 1;
resultFile = fopen(sprintf('%s%s_test.csv', resultDirectory, expTitle), 'w');
fprintf(resultFile, 'numInstanceCluster, numFeatureCluster, sigma, sigma2, lambda, gama, objectiveScore, accuracy, convergeTime\n');
load(sprintf('%sBestParameter_%s.mat', resultDirectory, expTitle));
PrepareGCMFExperiment;
main_GCMF_beta;
fclose(resultFile);