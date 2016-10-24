function [recall, precision] = getRecallPrecisionZeroShot(GroundTruth, ClusterResult, seedSet)
    [~, PredictionResult] = max(ClusterResult, [], 2);
    ClusterResult = zeros(size(ClusterResult,1), size(ClusterResult,2));
    for instanceId = 1: length(PredictionResult)
        ClusterResult(instanceId, PredictionResult(instanceId)) = 1;
    end
    % find the index of instance that supervised
    supervisedIndex = find(sum(GroundTruth, 2));
    % exclude seed when calculating performance
    supervisedIndex = setdiff(supervisedIndex, seedSet);
    ClusterResult = reshape(ClusterResult(supervisedIndex, :));
    GroundTruth = reshape(GroundTruth(supervisedIndex,:));
    numInstance = size(ClusterResult, 1);
    base = 0;
    overlap = 0;
    for i = 1:numInstance
        for j = 1:numInstance
            if j < i
                if ClusterResult(i, j) == 1
                    base = base + 1;
                    if GroundTruth(i, j) == 1
                        overlap = overlap + 1;
                    end
                end
            end
        end
    end
    precision = overlap/ base;
    
    base = 0;
    overlap = 0;
    for i = 1:numInstance
        for j = 1:numInstance
            if j < i
                if GroundTruth(i, j) == 1
                    base = base + 1;
                    if ClusterResult(i, j) == 1
                        overlap = overlap + 1;
                    end
                end
            end
        end
    end
    recall = overlap/ base;
end