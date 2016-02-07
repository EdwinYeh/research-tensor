SetParameterParallel;
exp_title = sprintf('ours_%d_allfree_cvx_sigma', datasetId);
sigma = 0.1;
lambda = 0.000001;

for parallelId = 1:5
    [initU, initV, initB, Lu, Label, TrueYMatrix] = prepareExperimentParallel(datasetId, numDom, parallelId, sigma);
    main_ours_allfree_cvx_parallel(exp_title, parallelId, lambda, initU, initV, initB, Lu, Label, TrueYMatrix);
end

main_ours_allfree_cvx;