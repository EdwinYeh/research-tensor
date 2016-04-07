SetParameter;
exp_title = 'DY_threshold';
resultFile = fopen(sprintf('../exp_result/%s_%d.csv', exp_title, datasetId), 'w');

sigmaScale = 10;

for sigmaOrder = 0: 6
    sigma = 10^(-3)* sigmaScale^sigmaOrder;
    PrepareExperiment;
    main_DY_threshold;
end

fclose(resultFile);