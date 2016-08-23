userIdList = [0];
userIdList = userIdList + 1;
numSeedSet = 100;

setup;
loadMturkDataset;

[seedDataMat] = loadSeed(userIdList, numSeedSet);

main_LPE;