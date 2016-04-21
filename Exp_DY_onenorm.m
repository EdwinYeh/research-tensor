SetParameter;
isTestPhase = true;
exp_title = sprintf('DY_onenorm_%d', datasetId);
resultFile = fopen(sprintf('../exp_result/%s.csv', exp_title), 'a');
fprintf(resultFile, 'sigma,sigma2,lambda,delta,objectiveScore,accuracy,trainingTime\n');
lambdaMaxOrder = 0;
deltaMaxOrder = 0;
randomTryTime = 5;
sigmaList = 0.05:0.05:0.05;
sigma2 = -1;
for tuneSigma = 1:length(sigmaList)
    sigma = sigmaList(tuneSigma);
    PrepareExperiment;
    for lambdaOrder = 0:lambdaMaxOrder
        lambda = 10^-4 * 100 ^ lambdaOrder;
        for deltaOrder = 0:deltaMaxOrder
            delta = 10^-4 * 10 ^ deltaOrder;
            main_DY_onenorm;
        end
        delta = 0;
        main_DY_onenorm;
    end
end
fclose(resultFile);
