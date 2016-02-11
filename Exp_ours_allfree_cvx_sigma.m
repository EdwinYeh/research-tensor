SetParameter;
exp_title = sprintf('ours_%d_allfree_cvx_sigma', datasetId);
isTestPhase = true;
lambdaTryTime = 3;
randomTryTime = 5;
sigmaList = [ 0.1, 0.5, 1, 10, 100];
for sigmaTryTime = 1:length(sigmaList)
    sigma = sigmaList(sigmaTryTime);
    PrepareExperiment2;
    for tuneLambda = 0:lambdaTryTime
        lambda = 0.00000001 * 100 ^ tuneLambda;
        showExperimentInfo(exp_title, datasetId, prefix, numSampleInstance, numSampleFeature, numInstanceCluster, numFeatureCluster, sigma);
            try
                    main_ours_allfree_cvx;
            catch exception
                    fprintf('Hyper parameter failed: sigma = %f, lambda = %f\n', sigma, lambda);
                    continue;
            end
    end
end