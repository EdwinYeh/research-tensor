function compareWithTheBestDX(newValidationAccuracy, newObjectiveScore, newTime, newSigma, newSigma2, newLambda, newGama, newDelta, newCpRank, newNumInstanceCluster, newNumFeatureCluster, resultDirectory, expTitle)
try
    load(sprintf('%sBestParameter_%s.mat', resultDirectory, expTitle));
    if newValidationAccuracy > bestValidationAccuracy
        disp('Update best record');
        bestValidationAccuracy = newValidationAccuracy;
        bestObjectiveScore = newObjectiveScore;
        bestTime = newTime;
        sigma = newSigma;
        sigma2 = newSigma2;
        lambda = newLambda;
        gama = newGama;
        delta = newDelta;
        cpRank = newCpRank;
        numInstanceCluster = newNumInstanceCluster;
        numFeatureCluster = newNumFeatureCluster;
        save(sprintf('%sBestParameter_%s.mat', resultDirectory, expTitle) , 'bestValidationAccuracy', 'bestObjectiveScore', 'bestTime', 'sigma', 'lambda', 'delta', 'cpRank', 'numInstanceCluster', 'numFeatureCluster');
    end
catch
    disp('No past parameter record. Build one.')
    bestValidationAccuracy = newValidationAccuracy;
        bestObjectiveScore = newObjectiveScore;
        bestTime = newTime;
        sigma = newSigma;
        sigma2 = newSigma2;
        lambda = newLambda;
        gama = newGama;
        delta = newDelta;
        cpRank = newCpRank;
        numInstanceCluster = newNumInstanceCluster;
        numFeatureCluster = newNumFeatureCluster;
    save(sprintf('%sBestParameter_%s.mat', resultDirectory, expTitle) , 'bestValidationAccuracy', 'bestObjectiveScore', 'bestTime', 'sigma', 'lambda', 'delta', 'cpRank', 'numInstanceCluster', 'numFeatureCluster');
end
end