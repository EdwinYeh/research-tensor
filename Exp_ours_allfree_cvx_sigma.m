SetParameter;
exp_title = sprintf('ours_%d_allfree_cvx_sigma', datasetId);
numSuccessParameter = 0;
bestValidateAccuracy = 0;
bestLambda = 0;
bestSigma = 0;
lambdaTryTime = 3;
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
            % if no error occur ++
            numSuccessParameter = numSuccessParameter + 1;
        if accuracy > bestValidateAccuracy
            bestValidateAccuracy = accuracy;
            bestLambda = lambda;
            bestSigma = sigma;
        end
    end
end

if numSuccessParameter > 0
    fprintf('Start testing\n');
    isTestPhase = true;
    %randomTryTime = 5;
    sigma = bestSigma;
    lambda = bestLambda;
    PrepareExperiment2;
    exp_title = sprintf('ours_%d_allfree_cvx_sigma', datasetId);
    showExperimentInfo(exp_title, datasetId, prefix, numSampleInstance, numSampleFeature, numInstanceCluster, numFeatureCluster, sigma);
    try
        main_ours_allfree_cvx;
    catch exception
        fprintf(resultFile, 'Parameter failed in test phase\n');
    end
else
    fprintf('No parameter is useful\n');
end