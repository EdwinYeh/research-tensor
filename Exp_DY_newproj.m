SetParameter;
isTestPhase = true;
exp_title = sprintf('DY_newproj_%d', datasetId);
lambdaTryTime = 10;
randomTryTime = 5;
sigmaList = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.45, 0.5];
sigma2 = -1;
delta = -1;
for sigmaTryTime = 1:length(sigmaList)
    sigma = sigmaList(sigmaTryTime);
    PrepareExperiment;
    for tuneLambda = 0:lambdaTryTime
        lambda = 0.00000001 * 10 ^ tuneLambda;
        main_DY_newproj;
    end
end