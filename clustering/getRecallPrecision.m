function [recall, precision] = getRecallPrecision(GroundTruth, Prediction, SeedSet)
% Calculate recall and precision for a domain

% Input
%   GroundTruth: instance x cluster matrix, has value 1 on (i, j) if instance
%       i belongs to j cluster
%   Prediction: instance x cluster matrix, prediction value from solver
%   supervisedIndex: vector that store the indeices that have groudtruth
% Output
%   Recall: matrix that stores recell of each cluster
%   Precision: vector that stores precision


GroundTruth(GroundTruth<0.5) = 0;
% soft-max
[~, PredictionResult] = max(Prediction, [], 2);
Prediction = zeros(size(Prediction,1), size(Prediction,2));
for instanceId = 1: length(PredictionResult)
    Prediction(instanceId, PredictionResult(instanceId)) = 1;
end

numCluster = size(GroundTruth, 2);
supervisedIndex = find(sum(GroundTruth, 2)>0);
supervisedIndex = setdiff(supervisedIndex, SeedSet);
GroundTruth = GroundTruth(supervisedIndex, :);
Prediction = Prediction(supervisedIndex, :);

precision = 0;
for clusterId = 1: numCluster;
    precisionDenominator = sum(Prediction(:, clusterId));
    if precisionDenominator == 0
        continue;
    end
    precisionIntersection = Prediction(:, clusterId) & GroundTruth(:, clusterId);
    precisionNumerator = sum(precisionIntersection);
    precision = precision + (precisionNumerator/ precisionDenominator);
end
precision = precision/ numCluster;

recall = 0;
for clusterId = 1: numCluster
    recallDenominator = sum(GroundTruth(:, clusterId));
    if recallDenominator == 0
        continue;
    end
    recallIntersection = Prediction(:, clusterId) & GroundTruth(:, clusterId);
    recallNumerator = sum(recallIntersection);
    recall = recall + (recallNumerator/ recallDenominator);
end
recall = recall/ numCluster;

if isnan(recall)
    recall = 0;
end

if isnan(precision)
    precision = 0;
end

end