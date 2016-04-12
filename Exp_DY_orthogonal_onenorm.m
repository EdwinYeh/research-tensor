SetParameter;
isTestPhase = true;
exp_title = sprintf('DY_orthogonal_onenorm_%d', datasetId);
lambdaTryTime = 10;
deltaTryTime = 9;
randomTryTime = 5;
sigmaList = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.45, 0.5];
for sigmaTryTime = 1:length(sigmaList)
    sigma = sigmaList(sigmaTryTime);
    PrepareExperiment;
    for lambdaOrder = 0:lambdaTryTime
        for deltaOrder = 0:deltaTryTime
            delta = 0.000000001 * 10 ^ deltaOrder;
            lambda = 0.00000001 * 10 ^ lambdaOrder;
            main_DY_orthogonal_onenorm;
        end
    end
end
