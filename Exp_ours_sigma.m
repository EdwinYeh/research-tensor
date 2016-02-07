SetParameter;
exp_title = sprintf('ours_%d_sigma', datasetId);
bestValidateAccuracy = 0;
bestLambda = 0;
bestSigma = 0;
lambdaTryTime = 0;
sigmaList = [0.1, 0.2, 0.3, 0.4, 0.5];
for sigmaTryTime = 1:length(sigmaList)
    sigma = sigmaList(sigmaTryTime);
    PrepareExperiment;
    for tuneLambda = 0:lambdaTryTime
        lambda = 0.000001 * 10 ^ tuneLambda;
        showExperimentInfo(exp_title, datasetId, prefix, numSampleInstance, numSampleFeature, numInstanceCluster, numFeatureCluster, sigma);
        main_ours;
        if accuracy > bestValidateAccuracy
            bestValidateAccuracy = accuracy;
            bestLambda = lambda;
            bestSigma = sigma;
        end
    end
end

fprintf('Start testing\n');
SetParameter;
isTestPhase = true;
randomTryTime = 1;
sigma = bestSigma;
lambda = bestLambda;
PrepareExperiment;
exp_title = sprintf('ours_%d_sigma', datasetId);
showExperimentInfo(exp_title, datasetId, prefix, numSampleInstance, numSampleFeature, numInstanceCluster, numFeatureCluster, sigma);
main_ours;