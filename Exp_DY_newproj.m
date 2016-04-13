SetParameter;
isTestPhase = true;
exp_title = sprintf('DY_newproj_%d', datasetId);
lambdaTryTime = 0;
randomTryTime = 1;
sigmaList = [0.05];
delta = -1;
sigma2 = -1;
for sigmaTryTime = 1:length(sigmaList)
    sigma = sigmaList(sigmaTryTime);
    PrepareExperiment;
    for tuneLambda = 0:lambdaTryTime
        lambda = 0.00000001 * 10 ^ tuneLambda;
        main_DY_newproj;
    end
end