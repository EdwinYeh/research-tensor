SetParameter;
exp_title = 'DY_fro_norm_H';

sigmaScale = 10;
sigma2Scale = 10;

for sigmaOrder = 0: 12
    sigma = 10^(-6)* sigmaScale^sigmaOrder;
    for sigma2Order = 0: 12
        sigma2 = 10^(-6)* sigma2Scale^sigma2Order;
        prepareDYExperiment;
        main_DY_fro_norm_H;
    end
end
