SetParameter;
isTestPhase = true;
exp_title = sprintf('DY_onenorm_%d', datasetId);
lambdaMaxOrder = 10;
deltaMaxOrder = 10;
randomTryTime = 5;
sigmaList = 0.05:0.05:0.5;
sigma2 = -1;
for sigmaTryTime = 1:length(sigmaList)
    sigma = sigmaList(sigmaTryTime);
    PrepareExperiment;
    for lambdaOrder = 0:lambdaMaxOrder
    lambda = 0.00000001 * 10 ^ lambdaOrder;
        for deltaOrder = 0:deltaMaxOrder
          delta = 0.000000001 * 10 ^ deltaOrder;
          main_DY_onenorm;
        end
        delta = 0;
        main_DY_onenorm;
    end
end
