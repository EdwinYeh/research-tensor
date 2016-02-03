SetParameter;
exp_title = sprintf('ours_%d_allfree_cvx_toy_sigma', datasetId);
bestValidateAccuracy = 0;
bestLambda = 0;
bestSigma = 0;
lambdaTryTime = 3;
sigmaList = [0.1, 0.3, 0.5, 0.7, 0.9];
for sigmaTryTime = 1:1
    sigma = sigmaList(sigmaTryTime);
    PrepareExperiment2;
    for tuneLambda = 0:lambdaTryTime
        lambda = 0.000001 * 100 ^ tuneLambda;
        showExperimentInfo(exp_title, datasetId, prefix, numSampleInstance, numSampleFeature, numInstanceCluster, numFeatureCluster, sigma);
        main_ours_allfree_cvx;
        if accuracy > bestValidateAccuracy
            bestValidateAccuracy = accuracy;
            bestLambda = lambda;
            bestSigma = sigma;
        end
    end
end

isTestPhase = true;
randomTryTime = 1;
sigma = bestSigma;
lambda = bestLambda;
PrepareExperiment2;
exp_title = sprintf('ours_%d_allfree_cvx_toy_sigma', datasetId);
showExperimentInfo(exp_title, datasetId, prefix, numSampleInstance, numSampleFeature, numInstanceCluster, numFeatureCluster, sigma);
main_ours_allfree_cvx;