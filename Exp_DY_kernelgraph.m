SetParameter;
isTestPhase = true;
exp_title = sprintf('DY_kernelgraph_%d', datasetId);
lambdaTryTime = 5;
sigmaTryTime = 10;
sigma2TryTime = 10;
randomTryTime = 2;
delta = -1;
for sigmaOrder = 1:sigmaTryTime
    sigma = (10^-5) * (10^sigmaOrder);
    for sigma2Order = 1:sigma2TryTime
        sigma2 = (10^-5) * (10^sigma2Order);
        prepareExperimentKernelGraph;
        for tuneLambda = 0:lambdaTryTime
            lambda = 0.00000001 * 100 ^ tuneLambda;
            main_DY_newproj;
        end
    end
end
