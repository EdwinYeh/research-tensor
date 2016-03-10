SetParameter;
exp_title = sprintf('ours_%d_allfree_cvx_rY_hyper', datasetId);
isTestPhase = true;
lambdaTryTime = 0;
omegaTryTime = 6;
randomTryTime = 1;
sigmaList = [ 0.1];
for sigmaTryTime = 1:length(sigmaList)
    sigma = sigmaList(sigmaTryTime);
    PrepareExperiment2;
    for tuneLambda = 0:lambdaTryTime
        lambda = 0.00000001 * 100 ^ tuneLambda;
        for tuneOmega = 0:omegaTryTime
            omega = 0.000001 * 100 ^ tuneOmega;
            showExperimentInfo(exp_title, datasetId, prefix, numSampleInstance, numSampleFeature, numInstanceCluster, numFeatureCluster, sigma);
            % try
            main_ours_allfree_cvx_rY;
            % catch exception
            %  fprintf('Hyper parameter failed: sigma = %f, lambda = %f, delta = %f, omega = %f\n', sigma, lambda, delta, omega);
            %disp(exception);
            %continue;
            %end
        end
    end
end