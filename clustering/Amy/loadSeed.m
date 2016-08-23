function [AllSeedData] = loadSeed( userIdList, numSeedSet )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
AllSeedData = cell(length(userIdList), numSeedSet);
for userId = userIdList
    SeedData = load(sprintf('datasets/SeedData_mturk_%d.mat', userId-1));
    for seedSetId = 1:numSeedSet
        AllSeedData{userId, seedSetId} = SeedData.SeedCluster{seedSetId};
    end
end

end

