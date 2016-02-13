SetParameter;
exp_title = sprintf('ours_%d_allfree_cvx_DBLP', datasetId);
isTestPhase = true;
lambdaTryTime = 5;
randomTryTime = 1;
PrepareExperiment_DBLP;

for tuneLambda = 0:lambdaTryTime
    lambda = 0.0000000001 * 100 ^ tuneLambda;
    showExperimentInfo(exp_title, datasetId, prefix, numSampleInstance, numSampleFeature, numInstanceCluster, numFeatureCluster, sigma);
    try
        main_ours_allfree_cvx;
    catch exception
        fprintf(resultFile, '%f, %f, failed, %s\n', sigma, lambda, exception.message);
        continue;
    end
end
