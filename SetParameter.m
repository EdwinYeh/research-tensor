exp_title = sprintf('Motar2_W_%d', datasetId);
isUpdateAE = true;
isSampleInstance = true;
isSampleFeature = true;
isRandom = true;
isUsingTensor = true;
numSampleInstance = [0, 0];
numSampleFeature = 2000;
maxIter = 100;
lambdaTryTime = 3;
randomTryTime = 1;
numDom = 2;
sourceDomain = 1;
targetDomain = 2;

if datasetId <= 6
    dataType = 1;
    prefix = '../20-newsgroup/';
    numInstanceCluster = 4;
    numFeatureCluster = 5;
    numClass = [2, 2];
    sigma = 1;
elseif datasetId > 6 && datasetId <=9
    dataType = 1;
    prefix = '../Reuter/';
    numInstanceCluster = 4;
    numFeatureCluster = 5;
    numClass = [2, 2];
    sigma = 0.1;
elseif datasetId == 10
    dataType = 2;
    prefix = '../Animal_img/';
    numInstanceCluster = 4;
    numFeatureCluster = 5;
    numClass = [2, 2];
    sigma = 0.1;
end

domainNameList = {sprintf('source%d.csv', datasetId), sprintf('target%d.csv', datasetId)};

numSourceInstanceList = [3913 3906 3782 3953 3829 3822 1237 1016 897 4460 60];
numTargetInstanceList = [3925 3909 3338 3960 3389 3373 1207 1043 897 4601 30];
numSourceFeatureList = [57309 59463 60800 58463 60800 60800 4771 4415 4563 4940 26];
numTargetFeatureList = [57913 59474 61188 59474 61188 61188 4771 4415 4563 4940 26];

alpha = 0;
beta = 0;