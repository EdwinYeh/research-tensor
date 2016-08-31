function [X, Y, XW, Su, Du, AllSeedCluster, AllPerceptionSeedFilter, AllSeedSet] = ...
    prepareExperimentMturk(datasetName, userIdList, sigmaInstsnce, maxSeedCombination)
% Input:
%   userIdArray: 1-d array saving userId involved in experiment
% Output:
%   X: low level feature matrix of instance (#instance * #feature)
%   Y: perception feature matrix of instance (#instance * #perception feature)
%   XW: cluster indicator matrix of instance (#instance * #cluster)
%   Su, Du: for calculating Laplacian matrix of instance
%   Sv, Dv: for calculating Laplacian matrix of perception feature
%   SeedFilter: filter matrix that has value 1 on seed positions and remains 0 otherwise
% Note:
%   (1) #feature is shared to all users
%   (2) #perception feature & #cluster are different from users
numDom = length(userIdList);
X = cell(1,numDom);
Y = cell(1, numDom);
XW = cell(1, numDom);
Su = cell(1, numDom);
Du = cell(1, numDom);
SeedSet = cell(maxSeedCombination,1);
SeedCluster = cell(maxSeedCombination,1);
PerceptionSeedFilter = cell(maxSeedCombination,1);
AllSeedSet = cell(maxSeedCombination, numDom);
AllSeedCluster = cell(maxSeedCombination, numDom);
AllPerceptionSeedFilter = cell(maxSeedCombination, numDom);

%     Sv = cell(1, domNum);
%     Dv = cell(1, domNum);
for domId = 1:numDom;
    userId = userIdList(domId);
    load(sprintf('../../mturk/User%d.mat', userId));
    load('../../mturk/data_feature.mat');
    data_feature = normr(data_feature);
    X{domId} = data_feature;
    Y{domId} = PerceptionInstance;
    XW{domId} = InstanceCluster;
    [numInstance, ~] = size(data_feature);
    [numPerception, ~] = size(PerceptionInstance);
    
    Su{domId} = zeros(numInstance, numInstance);
    Du{domId} = zeros(numInstance, numInstance);
    Su{domId} = gaussianSimilarityMatrix(data_feature, sigmaInstsnce);
    Su{domId}(isnan(Su{domId})) = 0;
    for instanceId = 1:numInstance
        Du{domId}(instanceId,instanceId) = sum(Su{domId}(instanceId,:));
    end
    
    for seedCombination = 1: maxSeedCombination
        try
            SeedData = load(sprintf('SeedData_%s_%d.mat', datasetName, userId));
            SeedSet = SeedData.SeedSet;
            SeedCluster = SeedData.SeedCluster;
            PerceptionSeedFilter = SeedData.PerceptionSeedFilter;
        catch
            disp('SeedData.mat doesnt exist create a new one');
            [SeedSet{seedCombination}, SeedCluster{seedCombination}, PerceptionSeedFilter{seedCombination}] = generateSeed(InstanceCluster, numPerception, 2);
        end
    end
    save(sprintf('SeedData_%s_%d.mat', datasetName, userId), 'SeedSet', 'SeedCluster', 'PerceptionSeedFilter');
    for seedCombination = 1: maxSeedCombination
        AllSeedCluster{seedCombination, domId} = SeedCluster{seedCombination};
        AllSeedSet{seedCombination, domId} = SeedSet{seedCombination};
        AllPerceptionSeedFilter{seedCombination, domId} = PerceptionSeedFilter{seedCombination};
    end
end
end

function [SeedSet, SeedCluster, PerceptionSeedFilter] = generateSeed(InstanceCluster, numPerception, seedCount)
[numInstance, numCluster] = size(InstanceCluster);
SeedCluster = zeros(numInstance, numCluster);
PerceptionSeedFilter = zeros(numPerception, numInstance);
SeedSet = zeros(numCluster*seedCount, 1);
for clusterId = 1: numCluster
    % Find each cluster's random two instances to be the cluster's seeds
    seed = find(InstanceCluster(:, clusterId));
    seed = seed(randperm(length(seed)));
    seed = seed(1:seedCount);
    SeedSet(clusterId*seedCount-(seedCount-1): clusterId*seedCount) = seed;
    for seedId = 1: seedCount
        SeedCluster(seed(seedId), clusterId) = 1;
        PerceptionSeedFilter(:, seed(seedId)) = 1;
    end
end
end