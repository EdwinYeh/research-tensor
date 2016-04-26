% Set default parameter (can be modified later)
isUpdateAE = true;
isSampleInstance = true;
isSampleFeature = true;
isRandom = true;
isUsingTensor = true;
isTestPhase = false;

numSampleFeature = 2000;
sigma = 0.1;
sigma2 = 0.1;
cpRank = 50;
numInstanceCluster = 50;
numFeatureCluster = 50;
numClass = [2, 2];

maxIter = 1000;
numCVFold = 5;
randomTryTime = 1;
numDom = 2;
sourceDomain = 1;
targetDomain = 2;
alpha = 0;
beta = 0;
