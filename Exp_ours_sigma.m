SetParameter;
isTestPhase = true;
exp_title = sprintf('ours_%d_sigma', datasetId);
lambdaTryTime = 5;
randomTryTime = 1;
sigmaList = [0.1];
for sigmaTryTime = 1:length(sigmaList)
    sigma = sigmaList(sigmaTryTime);
    PrepareExperiment;
    for tuneLambda = 0:lambdaTryTime
        lambda = 0.00000001 * 100 ^ tuneLambda;
        showExperimentInfo(exp_title, datasetId, prefix, numSampleInstance, numSampleFeature, numInstanceCluster, numFeatureCluster, sigma);
        main_ours;
    end
end