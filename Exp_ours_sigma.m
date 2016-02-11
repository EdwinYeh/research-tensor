SetParameter;
isTestPhase = true;
exp_title = sprintf('ours_%d_sigma', datasetId);
lambdaTryTime = 3;
sigmaList = [0.1,  0.5, 1, 10, 100];
for sigmaTryTime = 1:length(sigmaList)
    sigma = sigmaList(sigmaTryTime);
    PrepareExperiment;
    for tuneLambda = 0:lambdaTryTime
        lambda = 0.00000001 * 100 ^ tuneLambda;
        showExperimentInfo(exp_title, datasetId, prefix, numSampleInstance, numSampleFeature, numInstanceCluster, numFeatureCluster, sigma);
        main_ours;
    end
end