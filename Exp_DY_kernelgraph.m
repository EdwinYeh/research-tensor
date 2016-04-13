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
        prepareExperimentKernelGraph;
        for lambdaOrder = 0:lambdaMaxOrder
            lambda = 0.00000001 * 10 ^ lambdaOrder;
            main_DY_onenorm;
        end
    end
end
