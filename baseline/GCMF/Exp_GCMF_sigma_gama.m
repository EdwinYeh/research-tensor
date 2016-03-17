SetParameter;
isTestPhase = true;
exp_title = sprintf('ours_%d_GCMF_sigma', datasetId);
lambdaTryTime = 3;
gamaTryTime = 3;
randomTryTime = 5;
sigmaList = [0.1];
for sigmaTryTime = 1:length(sigmaList)
    sigma = sigmaList(sigmaTryTime);
    for tuneGama = 0:gamaTryTime;
        gama  = 0.000001 * 100 ^ tuneGama;
        PrepareExperiment;
        for tuneLambda = 0:lambdaTryTime
            lambda = 0.000001 * 100 ^ tuneLambda;
            showExperimentInfo(exp_title, datasetId, prefix, numSampleInstance, numSampleFeature, numInstanceCluster, numFeatureCluster, sigma);
            main_GCMF_advanced;
        end
    end
end
