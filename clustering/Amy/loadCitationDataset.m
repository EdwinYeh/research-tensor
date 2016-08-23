% citationDataset
datasetName = 'citationDataset';
load([parentPath inputPath datasetName '.mat']);
X = X';% transform to d*n (based on the objective)
allP = cluster_supervision;
allP = allP';
cluster_userIdx = cluster_userIndex; % k*1 (user index starts from 0)
cluster_data = groundTrue_cluster_data; %k*n



