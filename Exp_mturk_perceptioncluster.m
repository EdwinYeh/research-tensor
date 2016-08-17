function Exp_mturk_perceptioncluster(userIdList)
resultDirectory = '../../exp_result/Mturk/';
mkdir(resultDirectory);
domainNum = length(userIdList);
expTitle = 'Mturk';
for domId = 1:domainNum
    expTitle = [expTitle '_' num2str(userIdList(domId))];
end
resultFile = fopen(sprintf('%s%s.csv', resultDirectory, expTitle), 'a');
fprintf(resultFile, 'sigma,cpRank,perceptionCluster,lambda,gama,precisionAvg,recallVarAvg,objective,trainingTime\n');

gamaStart = 10^-6;
gamaScale = 10^2;
gamaMaxOrder = 5;

lambdaStart = 10^-12;
lambdaScale = 10^2;
lambdaMaxOrder = 5;

sigmaList = 0.75:0.5:5.25;
perceptionClusterNumList = [15, 30];
cpRankList = [30, 50];

maxRandomTryTime = 5;
maxSeedCombination = 3;

for tuneSigma = 1:length(sigmaList)
    sigma = sigmaList(tuneSigma);
    [X, Y, XW, Su, Du, SeedCluster, PerceptionSeedFilter, SeedSet] = prepareExperimentMturk(expTitle, userIdList, sigma, maxSeedCombination);
    for tuneCPRank = 1: length(cpRankList)
        cpRank = cpRankList(tuneCPRank);
        for tunePerceptionClusterNum = 1: length(perceptionClusterNumList)
            perceptionClusterNum = perceptionClusterNumList(tunePerceptionClusterNum);
            if perceptionClusterNum <=cpRank
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
                                    % fprintf('validation index domain%d: %d~%d\n', domID, min(validationIndex{domID}), max(validationIndex{domID}));
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
                                hyperparam.perceptionClusterNum = perceptionClusterNum;
                                
                                trainingTimer = tic;
                                output=solver_orthognal_perceptioncluster(input, hyperparam);
                                trainingTime = toc(trainingTimer);
                                randomTrainingTime(randomTryTime) = trainingTime;
                                
                                % Recall matrix for each domain
                                Recall = cell(1, domainNum);
                                % Recall varience of each domain
                                VarRecall = zeros(1, domainNum);
                                % Precision of each domain
                                Precision = zeros(1, domainNum);
                                for domId = 1: domainNum
                                    [~, Precision(domId)] = getRecallPrecision(XW{domId}, output.XW{domId}, SeedSet{domId});
                                    VarRecall(domId) = var(Recall{domId});
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
                        fprintf(resultFile, '%g,%g,%g,%g,%g,%g,%g,%g\n', sigma, cpRank, perceptionClusterNum, lambda, gama, avgSeedCombinationPrecision, avgSeedCombinationObjective, avgSeedCombinationTrainingTime);
                        fprintf('%g,%g,%g,%g,%g,%g,%g,%g\n', sigma, cpRank, perceptionClusterNum, lambda, gama, avgSeedCombinationPrecision, avgSeedCombinationObjective, avgSeedCombinationTrainingTime);
                    end
                end
            end
        end
    end
end
fclose(resultFile);
end