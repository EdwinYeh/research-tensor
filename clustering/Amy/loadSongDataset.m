% songDataset
datasetName = 'songDataset';
load([parentPath inputPath datasetName '.mat']);
X = X';% transform to d*n (based on the objective)
allP = cluster_supervision;
allP = allP';
cluster_userIdx = cluster_userId; % k*1 (user index starts from 0)
cluster_userIndex = cluster_userId; 
cluster_data = cluster_data; %k*n



