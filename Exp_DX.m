% Please assign datasetId in the commend line

sigmaList = [0.05, 0.1, 0.4, 0.7, 1.0];
numInstanceClusterList = [5, 10, 20, 40];
numFeatureClusterList = [5, 10, 20, 40];
cpRankList = [5, 10, 20, 40];

directoryName = sprintf('../exp_result/DX/%d/', datasetId);
mkdir(directoryName);

for tuneCPRank = 1: length(cpRankList)
    cpRank = cpRankList(tuneCPRank);
    for tuneNumFeatureCluster = 1: length(numFeatureClusterList)
        numFeatureCluster = numFeatureClusterList(tuneNumFeatureCluster);
        for tuneNumInstanceCluster = 1: length(numInstanceClusterList)
            numInstanceCluster = numInstanceClusterList(tuneNumInstanceCluster);
            for tuneSigma = 1: length(sigmaList)
                sigma = sigmaList(tuneSigma);
                prepareDXExperiment;
                main_DX_new_project;
            end
        end
    end
end