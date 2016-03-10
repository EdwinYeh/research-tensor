SetParameter;
isTestPhase = true;
exp_title = sprintf('ours_%d_GCMF_sigma', datasetId);
lambdaTryTime = 0;
randomTryTime = 1;
sigmaList = [0.1];
gamaList = [0.1];
for sigmaTryTime = 1:length(sigmaList)
    for gamaTryTime = 1:length(gamaList);
        sigma = sigmaList(sigmaTryTime);
        gama  = gamaList(gamaTryTime);
        PrepareExperiment;
        for tuneLambda = 0:lambdaTryTime
            lambda = 0.000001 * 10 ^ tuneLambda;
            showExperimentInfo(exp_title, datasetId, prefix, numSampleInstance, numSampleFeature, numInstanceCluster, numFeatureCluster, sigma);
            main_GCMF;
        end
    end
end