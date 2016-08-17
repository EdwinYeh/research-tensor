function Exp_mturk(userIdList)
resultDirectory = '../../exp_result/Mturk/';
mkdir(resultDirectory);
domainNum = length(userIdList);
expTitle = 'Mturk';
for domId = 1:domainNum
    expTitle = [expTitle '_' num2str(userIdList(domId))];
end
resultFile = fopen(sprintf('%s%s.csv', resultDirectory, expTitle), 'a');
fprintf(resultFile, 'sigma,cpRank,lambda,gama,avgPrecision,objective,trainingTime\n');

gamaStart = 10^-3;
gamaScale = 10^1;
gamaMaxOrder = 4;

lambdaStart = 10^-2;
lambdaScale = 10^1;
lambdaMaxOrder = 5;

sigmaList = 0.05:0.25:3;
cpRankList = [15, 40, 70];

maxRandomTryTime = 3;
maxSeedCombination = 3;

for tuneSigma = 1:length(sigmaList)
    sigma = sigmaList(tuneSigma);
    [X, Y, XW, Su, Du, SeedCluster, PerceptionSeedFilter, SeedSet] = ...
        prepareExperimentMturk(expTitle, userIdList, sigma, maxSeedCombination);
    for tuneCPRank = 1: length(cpRankList)
        cpRank = cpRankList(tuneCPRank);
        for lambdaOrder = 0: lambdaMaxOrder
            lambda = lambdaStart * lambdaScale ^ lambdaOrder;
            for gamaOrder = 0: gamaMaxOrder
                gama = gamaStart * gamaScale ^ gamaOrder;
                seedCombinationPrecision = zeros(1, maxSeedCombination);
                seedCombinationTrainingTime = zeros(1, maxSeedCombination);
                seedCombinationObjective = zeros(1, maxSeedCombination);
                for seedCombination = 1: maxSeedCombination
                    randomPrecision = zeros(1, maxRandomTryTime);
                    randomTrainingTime = zeros(1, maxRandomTryTime);
                    randomObjective = zeros(1, maxRandomTryTime);
                    for randomTryTime = 1:maxRandomTryTime
                        for domId = 1: domainNum
                            input.S{domId}=PerceptionSeedFilter{seedCombination, domId};
                            input.SeedSet{domId} = SeedSet{seedCombination, domId};
                            input.SeedCluster{domId}=SeedCluster{seedCombination, domId};
                            input.X{domId} = X{domId};
                            % If Y has all 0 row or col update rule will fail
                            Y{domId}(Y{domId}==0) = 10^-18;
                            input.Y{domId} = Y{domId};
                            input.XW{domId} = XW{domId};
                            input.Sxw{domId} = Su{domId};
                            input.Dxw{domId} = Du{domId};
                        end;
                        
                        hyperparam.beta = 0;
                        hyperparam.gamma = gama;
                        hyperparam.lambda = lambda;
                        hyperparam.cpRank = cpRank;                        
                        
                        trainingTimer = tic;
                        output=solver_orthognal(input, hyperparam);
                        trainingTime = toc(trainingTimer);
                        randomTrainingTime(randomTryTime) = trainingTime;
                                 
                        % Precision of each domain
                        Precision = zeros(1, domainNum);
                        for domId = 1: domainNum
                            [~, Precision(domId)] = getRecallPrecision(XW{domId}, output.XW{domId}, SeedSet{domId});
                        end
                        avgPrecision = mean(Precision);
                        randomPrecision(randomTryTime) = avgPrecision;
                        randomObjective(randomTryTime) = output.objective;
                    end
                    avgRandomPrecision = mean(randomPrecision);
                    avgRandomTrainingTime = mean(randomTrainingTime);
                    avgRandomObjective = mean(randomObjective);
                    seedCombinationPrecision(seedCombination) = avgRandomPrecision;
                    seedCombinationTrainingTime(seedCombination) = avgRandomTrainingTime;
                    seedCombinationObjective(seedCombination) = avgRandomObjective;
                end
                avgSeedCombinationPrecision = mean(seedCombinationPrecision);
                avgSeedCombinationTrainingTime = mean(seedCombinationTrainingTime);
                avgSeedCombinationObjective = mean(seedCombinationObjective);
                
                fprintf(resultFile, '%g,%g,%g,%g,%g,%g,%g\n', ...
                    sigma, cpRank, lambda, gama, avgSeedCombinationPrecision, avgSeedCombinationObjective, avgSeedCombinationTrainingTime);
                fprintf('%g,%g,%g,%g,%g,%g,%g\n', ...
                    sigma, cpRank, lambda, gama, avgSeedCombinationPrecision, avgSeedCombinationObjective, avgSeedCombinationTrainingTime);
            end
        end
    end
end
fclose(resultFile);
end