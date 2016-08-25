function [] = refineResult(datasetName, resultFileName)
    result = load(sprintf('../../../exp_result/%s/%s.mat', datasetName, resultFileName));
    numSeedSet = length(result.bestSeedCombinationPrecision);
    numUserId = length(result.bestSeedCombinationPrecision{1});
    fprintf('numSeedSet: %d, numUser: %d\n', numSeedSet, numUserId);
    PrecisionMatrix = zeros(numSeedSet, numUserId);
    RecallMatrix = zeros(numSeedSet, numUserId);
    FScoreMatrix = zeros(numSeedSet, numUserId);
    
    for seedSetId = 1: numSeedSet
        for userId = 1: numUserId
            PrecisionMatrix(seedSetId, userId) = result.bestSeedCombinationPrecision{seedSetId}(userId);
            RecallMatrix(seedSetId, userId) = result.bestSeedCombinationRecall{seedSetId}(userId);
            FScoreMatrix(seedSetId, userId) = result.bestSeedCombinationFScore{seedSetId}(userId);
        end
    end
    
    csvwrite(sprintf('Precision_%s.csv', resultFileName), PrecisionMatrix);
    csvwrite(sprintf('Recall_%s.csv', resultFileName), RecallMatrix);
    csvwrite(sprintf('FScore_%s.csv', resultFileName), FScoreMatrix);
end

