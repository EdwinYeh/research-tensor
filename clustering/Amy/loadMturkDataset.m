
% mturk dataset
datasetName = 'mturkDataset';
load([parentPath inputPath datasetName '.mat']);
cluster_userIdx = cluster_userIndex; % #all of the clusters * 1
allP = supervision_cluster;
cluster_data = cluster_data;
X = data_feature;
X = X';% transform to d*n (based on the objective)