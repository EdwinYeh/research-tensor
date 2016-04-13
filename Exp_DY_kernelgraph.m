SetParameter;
isTestPhase = true;
exp_title = sprintf('DY_kernelgraph_%d', datasetId);
lambdaMaxOrder = 10;
sigmaMaxOrder = 15;
sigma2MaxOrder = 15;
randomTryTime = 5;
delta = 0;
for sigmaOrder = 1:sigmaMaxOrder
    sigma = (1) * (10^sigmaOrder);
    for sigma2Order = 1:sigma2MaxOrder
        sigma2 = (1) * (10^sigma2Order);
        PrepareExperimentKernelGraph;
        for tuneLambda = 0:lambdaTryTime
            lambda = 0.00000001 * 10 ^ tuneLambda;
            main_DY_newproj;
        end
    end
end
