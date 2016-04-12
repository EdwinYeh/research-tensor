% Please assign datasetId in the commend line
SetParameter;
sigmaList = [0.1];
numInstanceClusterList = [5];
numFeatureClusterList = [5];
cpRankList = [5];
expTitle = 'DX';

directoryName = sprintf('../exp_result/%s/%d/', expTitle, datasetId);
mkdir(directoryName);

for tuneSigma = 1: length(sigmaList)
    sigma = sigmaList(tuneSigma);
    for tuneCPRank = 1: length(cpRankList)
        cpRank = cpRankList(tuneCPRank);
        for tuneNumFeatureCluster = 1: length(numFeatureClusterList)
            numFeatureCluster = numFeatureClusterList(tuneNumFeatureCluster);
            for tuneNumInstanceCluster = 1: length(numInstanceClusterList)
                numInstanceCluster = numInstanceClusterList(tuneNumInstanceCluster);
                if numInstanceCluster <= cpRank && numFeatureCluster <= cpRank
%                     try
                        prepareDXExperiment;
                        main_DX;
%                     catch exception
%                         fprintf('%g,%g,%g,%d,%d,%d\n', lambda, gama, sigma, numInstanceCluster, numFeatureCluster, cpRank);
%                         disp(exception.message);
%                         continue;
%                     end
                end
            end
        end
    end
end
