function compareWithTheBest(newValidationAccuracy, newObjectiveScore, newTime, sigma, lambda, delta, cpRank, numInstanceCluster, numFeatureCluster, resultDirectory, expTitle)
try
    load(sprintf('%sBestParameter_%s.mat', resultDirectory, expTitle));
    if newValidationAccuracy > bestValidationAccuracy
        bestValidationAccuracy = newValidationAccuracy;
        bestObjectiveScore = newObjectiveScore;
        bestTime = newTime;
        save(sprintf('%sBestParameter_%s.mat', resultDirectory, expTitle) , 'bestValidationAccuracy', 'bestObjectiveScore', 'bestTime', 'sigma', 'lambda', 'delta', 'cpRank', 'numInstanceCluster', 'numFeatureCluster');
    end
catch
    bestValidationAccuracy = newValidationAccuracy;
    bestObjectiveScore = newObjectiveScore;
    bestTime = newTime;
    save(sprintf('%sBestParameter_%s.mat', resultDirectory, expTitle) , 'bestValidationAccuracy', 'bestObjectiveScore', 'bestTime', 'sigma', 'lambda', 'delta', 'cpRank', 'numInstanceCluster', 'numFeatureCluster');
end
end