function Exp_time_old(datasetName, userIdList, perceptionSeedRate, clusterSeedRate)
resultDirectory = sprintf('../../exp_result/%s/', datasetName);
parameterNameOrder = 'sigma, lambda, gama, cpRank';
mkdir(resultDirectory);
numDom = length(userIdList);
expTitle = [datasetName '(' num2str(perceptionSeedRate) ')' '(' num2str(clusterSeedRate) ')'];
for domId = 1:numDom
    expTitle = [expTitle '_' num2str(userIdList(domId))];
end
% resultFile = fopen(sprintf('%s%s.csv', resultDirectory, expTitle), 'a');
% fprintf(resultFile, 'sigma,cpRank,lambda,gama,avgPrecision,objective,trainingTime\n');

gamaStart = 10^-8;
gamaScale = 10^2;
gamaMaxOrder = 0;

lambdaStart = 10^-3;
lambdaScale = 10^2;
lambdaMaxOrder = 0;

if strcmp(datasetName, 'song')
    sigmaList = [300, 500, 700, 900];
else
    sigmaList = [0.005];
end
cpRankList = [80];

maxRandomTryTime = 1;
maxSeedCombination = 5;

bestParamPrecision = cell(1, maxSeedCombination);
bestParamRecall = cell(1, maxSeedCombination);
bestParamFScore = cell(1, maxSeedCombination);
bestParamTrainingTime = zeros(1, maxSeedCombination);
bestParamObjective = ones(1, maxSeedCombination)*Inf;
bestParamCombination = cell(1, maxSeedCombination);
Tracker = cell(1, maxSeedCombination);

for sigma = sigmaList
    [X, Y, XW, Su, Du, SeedCluster, SeedSet] = ...
        prepareExperiment(datasetName, userIdList, sigma, maxSeedCombination, clusterSeedRate);
    for cpRank = cpRankList
        for lambdaOrder = 0: lambdaMaxOrder
            lambda = lambdaStart * lambdaScale ^ lambdaOrder;
            for gamaOrder = 0: gamaMaxOrder
                gama = gamaStart * gamaScale ^ gamaOrder;
                fprintf('(sigma, lambda, gama, cpRank)=(%g,%g,%g,%g)\n', sigma, lambda, gama, cpRank);
                for seedCombinationId = 1: maxSeedCombination
                    RandomPrecision = zeros(maxRandomTryTime, numDom);
                    RandomRecall = zeros(maxRandomTryTime, numDom);
                    RandomFScore = zeros(maxRandomTryTime, numDom);
                    RandomTrainingTime = zeros(1, maxRandomTryTime);
                    RandomObjective = zeros(1, maxRandomTryTime);
                    for randomTryTime = 1:maxRandomTryTime
                        for domId = 1: numDom
%                             input.S{domId}=PerceptionSeedFilter{seedCombinationId, domId};
                            input.S{domId} = getRandomPerceptionSeed(Y{domId}, perceptionSeedRate);
                            input.SeedSet{domId} = SeedSet{seedCombinationId, domId};
                            input.SeedCluster{domId}=SeedCluster{seedCombinationId, domId};
                            % Set somain 1 to no cluster seed in zero shot experiment
%                             if isZeroShot && domId==1
%                                 input.SeedSet{domId} = [];
%                             end
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
                        output=solver_cp(input, hyperparam);
                        trainingTime = toc(trainingTimer);
                        RandomTrainingTime(randomTryTime) = trainingTime;
                        
                        iterFScore = zeros(20,1);
                        for iter = 1:20
                            if ~isempty(output.Tracker{1, 3}{iter})
                                iterU = output.Tracker{1, 3}{iter}{1};
                                [iterRecall, iterPrecision] = getRecallPrecision(XW{1}, iterU, SeedSet{seedCombinationId, 1});
                                iterFScore(iter) = 2*(iterRecall*iterPrecision)/(iterRecall+iterPrecision);
                            end
                        end
                        output.Tracker{4} = iterFScore;
                        Tracker{seedCombinationId} = output.Tracker;
                        save('time_old_Tracker.mat', 'Tracker');
                        
                        for domId = 1: numDom
%                             if isZeroShot
%                                 [RandomRecall(randomTryTime, domId), RandomPrecision(randomTryTime, domId)] =...
%                                     getRecallPrecisionZeroShot(XW{domId}, output.XW{domId}, SeedSet{seedCombinationId, domId});
%                             else
                                [RandomRecall(randomTryTime, domId), RandomPrecision(randomTryTime, domId)] =...
                                    getRecallPrecision(XW{domId}, output.XW{domId}, SeedSet{seedCombinationId, domId});
%                             end
                            RandomFScore(randomTryTime, domId) = 2*((RandomRecall(randomTryTime, domId)*RandomPrecision(randomTryTime, domId))/(RandomRecall(randomTryTime, domId)+RandomPrecision(randomTryTime, domId)));
                            if isnan(RandomFScore(randomTryTime, domId))
                                RandomFScore(randomTryTime, domId) = 0;
                            end
                        end
                        
                        % Precision of each domain
                        RandomObjective(randomTryTime) = output.objective;
                    end

                    [minRandomObjective, minObjRandomTime] = min(RandomObjective);
                    minRandomPrecision = RandomPrecision(minObjRandomTime, :);
                    minRandomRecall = RandomRecall(minObjRandomTime, :);
                    minRandomFScore = RandomFScore(minObjRandomTime, :);
                    minRandomTrainingTime = RandomTrainingTime(minObjRandomTime);

                    if minRandomObjective < bestParamObjective(seedCombinationId)
                        bestParamPrecision{seedCombinationId} = minRandomPrecision;
                        bestParamRecall{seedCombinationId} = minRandomRecall;
                        bestParamFScore{seedCombinationId} = minRandomFScore;
                        bestParamTrainingTime(seedCombinationId) = minRandomTrainingTime;
                        bestParamObjective(seedCombinationId) = minRandomObjective;
                        bestParamCombination{seedCombinationId} = [sigma, lambda, gama, cpRank];
                        save(sprintf('%s%s.mat', resultDirectory, expTitle), ...
                            'bestParamPrecision', 'bestParamRecall', 'bestParamFScore', ...
                            'bestParamTrainingTime', 'bestParamObjective', ...
                            'bestParamCombination', 'parameterNameOrder');
                    end
                end
            end
        end
    end
end
% fclose(resultFile);
end

function perceptionSeedFilter = getRandomPerceptionSeed(PerceptionInstance, perceptionSeedRate)
    [numPerception, numInstance] = size(PerceptionInstance);
    perceptionSeedFilter = zeros(numPerception, numInstance);
    supervisedIndex = find(PerceptionInstance);
    numSupervise = length(supervisedIndex);
    numPerceptionSeed = round(numSupervise* perceptionSeedRate);
    perceptionSeedIndex = supervisedIndex(randperm(numSupervise, numPerceptionSeed));
    perceptionSeedFilter(perceptionSeedIndex) = 1;
end

function NewGroundTruth = reshape(GroundTruth)
    [numInstance, numCluster] = size(GroundTruth);
    NewGroundTruth = zeros(numInstance, numInstance);
    for clusterId = 1:numCluster
        instanceInCluster = find(GroundTruth(:, clusterId) == 1);
        for i = 1: length(instanceInCluster)
            for j = 1:length(instanceInCluster)
                NewGroundTruth(instanceInCluster(i), instanceInCluster(j)) = 1;
            end
        end
    end
end

function [recall, precision] = getRecallPrecisionZeroShot(GroundTruth, ClusterResult, SeedSet)
    [~, PredictionResult] = max(ClusterResult, [], 2);
    ClusterResult = zeros(size(ClusterResult,1), size(ClusterResult,2));
    for instanceId = 1: length(PredictionResult)
        ClusterResult(instanceId, PredictionResult(instanceId)) = 1;
    end
    % find the index of instance that supervised
    supervisedIndex = find(sum(GroundTruth, 2));
    % find index of seed
    seedIndex = find(sum(SeedSet, 2));
    % exclude seed when calculating performance
    supervisedIndex = setdiff(supervisedIndex, seedIndex);
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