function [X, InstanceCluster, SeedPerception, SeedCluster, Su, Du, PerceptionSeedFilter, SeedClusterFilter, SeedSet] = prepareExperimentMturk2(expTitle, userIdList, sigmaInstsnce)
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
Su = cell(1, numDom);
Du = cell(1, numDom);
InstanceCluster = cell(1,numDom);
SeedSet = cell(1, numDom);
SeedCluster = cell(1, numDom);
SeedPerception = cell(1, numDom);
PerceptionSeedFilter = cell(1, numDom);
SeedClusterFilter = cell(1, numDom);
%     Sv = cell(1, domNum);
%     Dv = cell(1, domNum);
for domId = 1:numDom;
    userId = userIdList(domId);
    Data = load(sprintf('../mturk/User%d.mat', userId));
    load('../mturk/data_feature.mat');
    data_feature = normr(data_feature);
    X{domId} = data_feature;
    [numInstance, ~] = size(data_feature);
    [numPerception, ~] = size(Data.PerceptionInstance);
    InstanceCluster{domId} = Data.InstanceCluster;
    Su{domId} = zeros(numInstance, numInstance);
    Du{domId} = zeros(numInstance, numInstance);
    Su{domId} = gaussianSimilarityMatrix(data_feature, sigmaInstsnce);
    Su{domId}(isnan(Su{domId})) = 0;
    for userId = 1:numInstance
        Du{domId}(userId,userId) = sum(Su{domId}(userId,:));
    end
    
    try
        SeedData = load(sprintf('SeedData_%s.mat', expTitle));
        SeedSet = SeedData.SeedSet;
        SeedCluster = SeedData.SeedCluster;
        PerceptionSeedFilter = SeedData.PerceptionSeedFilter;
        SeedClusterFilter = SeedData.SeedClusterFilter;
    catch
        disp('SeedData.mat doesnt exist create a new one');
        [SeedSet{domId}, SeedCluster{domId}, PerceptionSeedFilter{domId}, SeedClusterFilter{domId}] = generateSeed(InstanceCluster{domId}, numPerception, 2);
    end
    SeedPerception{domId} = (Data.PerceptionInstance.*PerceptionSeedFilter{domId})';
end
save(sprintf('SeedData_%s.mat', expTitle), 'SeedSet', 'SeedCluster', 'PerceptionSeedFilter', 'SeedClusterFilter');
end

function [SeedSet, SeedCluster, PerceptionSeedFilter, SeedClusterFilter] = generateSeed(InstanceCluster, numPerception, seedCount)
[numInstance, numCluster] = size(InstanceCluster);
SeedCluster = zeros(numInstance, numCluster);
PerceptionSeedFilter = zeros(numPerception, numInstance);
SeedClusterFilter = zeros(numInstance, numCluster);
SeedSet = zeros(numCluster*seedCount, 1);
for clusterId = 1: numCluster
    % Find each cluster's random two instances to be the cluster's seeds
    seed = find(InstanceCluster(:, clusterId));
    seed = seed(randperm(length(seed)));
    seed = seed(1:seedCount);
    SeedSet(clusterId*seedCount-(seedCount-1): clusterId*seedCount) = seed;
    for seedId = 1: seedCount
        SeedCluster(seed(seedId), clusterId) = 1;
        SeedClusterFilter(seed(seedId), :) = 1;
        PerceptionSeedFilter(:, seed(seedId)) = 1;
    end
end
end