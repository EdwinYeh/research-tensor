SetParameter;
exp_title = 'DY_one_norm_H';
resultFile = fopen(sprintf('../exp_result/%s_%d.csv', exp_title, datasetId), 'w');

sigmaScale = 10;

for sigmaOrder = 0: 6
    sigma = 10^(-3)* sigmaScale^sigmaOrder;
    PrepareExperiment;
    main_DY_one_norm_H;
end

fclose(resultFile);