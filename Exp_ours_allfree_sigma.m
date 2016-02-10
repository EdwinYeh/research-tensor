SetParameter;
exp_title = sprintf('ours_%d_allfree_sigma', datasetId);
bestValidateAccuracy = 0;
bestLambda = 0;
bestSigma = 0;
lambdaTryTime = 3;
sigmaList = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
for sigmaTryTime = 1:length(sigmaList)
    sigma = sigmaList(sigmaTryTime);
    PrepareExperiment2;
    for tuneLambda = 0:lambdaTryTime
        lambda = 0.00000001 * 100 ^ tuneLambda;
        showExperimentInfo(exp_title, datasetId, prefix, numSampleInstance, numSampleFeature, numInstanceCluster, numFeatureCluster, sigma);
            try
                main_ours_allfree;
            catch exception
                continue;
            end
        if accuracy > bestValidateAccuracy
            bestValidateAccuracy = accuracy;
            bestLambda = lambda;
            bestSigma = sigma;
        end
    end
end

fprintf('Start testing\n');
isTestPhase = true;
randomTryTime = 5;
sigma = bestSigma;
lambda = bestLambda;
PrepareExperiment2;
exp_title = sprintf('ours_%d_allfree_cvx_sigma', datasetId);
showExperimentInfo(exp_title, datasetId, prefix, numSampleInstance, numSampleFeature, numInstanceCluster, numFeatureCluster, sigma);
main_ours_allfree;