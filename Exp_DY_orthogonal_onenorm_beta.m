SetParameter;
isTestPhase = true;
exp_title = sprintf('DY_orthogonal_onenorm_beta_%d', datasetId);
lambdaTryTime = 0;
deltaTryTime = 0;
randomTryTime = 1;
sigmaList = [0.5];
for sigmaTryTime = 1:length(sigmaList)
    sigma = sigmaList(sigmaTryTime);
    PrepareExperiment;
    for lambdaOrder = 0:lambdaTryTime
        for deltaOrder = 0:deltaTryTime
            delta = 0.000000001 * 10 ^ deltaOrder;
            lambda = 0.00000001 * 10 ^ lambdaOrder;
            main_DY_newproj_beta;
        end
    end
end
