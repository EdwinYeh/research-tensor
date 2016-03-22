SetParameter;
exp_title = sprintf('ours_%d_allfree_cvx_rH_hyper', datasetId);
isTestPhase = true;
lambdaTryTime = 0;
deltaTryTime = 6;
randomTryTime = 1;
sigmaList = [ 0.1];
for sigmaTryTime = 1:length(sigmaList)
    sigma = sigmaList(sigmaTryTime);
    PrepareExperiment2;
    for tuneLambda = 0:lambdaTryTime
        lambda = 0.00000001 * 100 ^ tuneLambda;
        for tuneDelta = 0:deltaTryTime
            delta = 0.000001 * 100 ^ tuneDelta;
            showExperimentInfo(exp_title, datasetId, prefix, numSampleInstance, numSampleFeature, numInstanceCluster, numFeatureCluster, sigma);
           try
                    main_ours_allfree_cvx_rH;
           catch exception
                    fprintf('Hyper parameter failed: sigma = %f, lambda = %f, delta = %f\n', sigma, lambda, delta);
                    disp(exception);
                    continue;
            end
        end
    end
end