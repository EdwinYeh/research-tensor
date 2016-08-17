function [ Recall, precision] = getRecallPrecision(GroundTruth, Prediction, SeedSet)
% Calculate recall and precision for a domain

% Input
%   GroundTruth: instance x cluster matrix, has value 1 on (i, j) if instance
%       i belongs to j cluster
%   Prediction: instance x cluster matrix, prediction value from solver
%   supervisedIndex: vector that store the indeices that have groudtruth
% Output
%   Recall: matrix that stores recell of each cluster
%   Precision: vector that stores precision
    supervisedIndex = find(sum(GroundTruth, 2));
%     supervisedIndex = setdiff(supervisedIndex, SeedSet); 
    GroundTruth = GroundTruth(supervisedIndex, :);
    Prediction = Prediction(supervisedIndex, :);
    [~, PredictionResult] = max(Prediction, [], 2);
    Prediction = zeros(size(Prediction,1), size(Prediction,2));
    for instanceId = 1: length(PredictionResult)
        Prediction(instanceId, PredictionResult(instanceId)) = 1;
    end
   
    PrecisionIntersect = Prediction & GroundTruth;
    precisionNumerator = sum(sum(PrecisionIntersect));
    precisionDenominator = sum(sum(Prediction));
    precision = precisionNumerator/ precisionDenominator;
    
    clusterNum = size(GroundTruth, 2);
    Recall = zeros(1, clusterNum);
    for clusterId = 1: clusterNum
        RecallIntersect = Prediction(:, clusterId) & GroundTruth(:, clusterId);
        recallNumerator = sum(RecallIntersect);
        recallDenominator = sum(GroundTruth(:, clusterId));
        Recall(clusterId) = recallNumerator/ recallDenominator;
    end
end

