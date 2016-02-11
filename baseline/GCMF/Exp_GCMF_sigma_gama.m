SetParameter;
exp_title = sprintf('ours_%d_GCMF_sigma', datasetId);
bestValidateAccuracy = 0;
bestLambda = 0;
bestSigma = 0;
lambdaTryTime = 3;
sigmaList = [0.1, 0.5, 1, 10, 100];
gamaList = [0.1, 0.5, 1, 10, 100];
for sigmaTryTime = 1:length(sigmaList)
    for gamaTryTime = 1:length(gamaList);
        sigma = sigmaList(sigmaTryTime);
        gama  = gamaList(gamaTryTime);
        PrepareExperiment;
        for tuneLambda = 0:lambdaTryTime
            lambda = 0.000001 * 10 ^ tuneLambda;
            showExperimentInfo(exp_title, datasetId, prefix, numSampleInstance, numSampleFeature, numInstanceCluster, numFeatureCluster, sigma);
            main_GCMF;
            if accuracy > bestValidateAccuracy
                bestValidateAccuracy = accuracy;
                bestLambda = lambda;
                bestSigma = sigma;
            end
        end
    end
end

fprintf('Start testing\n');
isTestPhase = true;
randomTryTime = 5;
sigma = bestSigma;
lambda = bestLambda;
PrepareExperiment;
exp_title = sprintf('ours_%d_allfree_cvx_sigma', datasetId);
showExperimentInfo(exp_title, datasetId, prefix, numSampleInstance, numSampleFeature, numInstanceCluster, numFeatureCluster, sigma);
main_GCMF;