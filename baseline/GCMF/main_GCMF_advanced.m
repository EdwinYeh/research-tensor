disp('Start training');

% if isTestPhase
%     resultFile = fopen(sprintf('result_%s.csv', exp_title), 'a');
%     fprintf(resultFile, 'sigma,gama,lambda,objectiveScore,accuracy,trainingTime\n');
% end
directoryName = sprintf('../../../exp_result/GCMF/%d/', datasetId);
mkdir(directoryName);
fprintf('Use Lambda: %f, Gama: %f\n', lambda, gama);
bestObjectiveScore = Inf;

for t = 1: randomTryTime
    TotalTimer = tic;
    %re-initialize
    U = initU(t, :);
    V = initV(t, :);
    H = initH{t};
   
    HChildCell = cell(1, numDom);
    HMotherCell = cell(1, numDom);
    
    %Iterative update
    newObjectiveScore = Inf;
    oldObjectiveScore = Inf;
    iter = 0;
    diff = Inf;
    
    while (diff >= 0.001  && iter < maxIter)%(abs(ObjectiveScore - newObjectiveScore) >= 0.1 && iter < maxIter)
        iter = iter + 1;
        oldObjectiveScore = newObjectiveScore;
        %disp(sprintf('\t#Iterator:%d', iter));
        %disp(newObjectiveScore);
        newObjectiveScore = 0;
        for i = 1:numDom
            %disp(sprintf('\t\tupdate V...'));
            %update V
            V{i} = V{i}.*sqrt((X{i}'*U{i}*H+gama*Sv{i}*V{i})./(V{i}*H'*U{i}'*U{i}*H+gama*Dv{i}*V{i}));
            V{i}(isnan(V{i})) = 0;
            V{i}(~isfinite(V{i})) = 0;
            %col normalize
            [r, ~] = size(V{i});
            for tmpI = 1:r
                bot = sum(abs(V{i}(tmpI,:)));
                if bot == 0
                    bot = 1;
                end
                V{i}(tmpI,:) = V{i}(tmpI,:)/bot;
            end
            V{i}(isnan(V{i})) = 0;
            V{i}(~isfinite(V{i})) = 0;
            
            %disp(sprintf('\t\tupdate U...'));
            %update U
            U{i} = U{i}.*sqrt((X{i}*V{i}*H'+lambda*Su{i}*U{i})./(U{i}*H*V{i}'*V{i}*H'+lambda*Du{i}*U{i}));
            U{i}(isnan(U{i})) = 0;
            U{i}(~isfinite(U{i})) = 0;
            %col normalize
            [r, ~] = size(U{i});
            for tmpI = 1:r
                bot = sum(abs(U{i}(tmpI,:)));
                if bot == 0
                    bot = 1;
                end
                U{i}(tmpI,:) = U{i}(tmpI,:)/bot;
            end
            U{i}(isnan(U{i})) = 0;
            U{i}(~isfinite(U{i})) = 0;
            
            %update H
            HChild = zeros(numInstanceCluster, numFeatureCluster);
            HMother = zeros(numInstanceCluster, numFeatureCluster);
            for j = 1:numDom
                HChildCell{j} = U{j}'*X{j}*V{j};
                HMotherCell{j} = U{j}'*U{j}*H*V{j}'*V{j};
            end
            for j = 1:numDom
                HChild = HChild + HChildCell{j};
                HMother = HMother + HMotherCell{j};
            end
            H = H.*sqrt(HChild./HMother);
        end
        %disp(sprintf('\tCalculate this iterator error'));
        for i = 1:numDom
            result = U{i}*H*V{i}';
            normEmp = norm((X{i} - result), 'fro')*norm((X{i} - result), 'fro');
            smoothU = lambda*trace(U{i}'*Lu{i}*U{i});
            smoothV = gama*trace(V{i}'*Lv{i}*V{i});
            loss = normEmp + smoothU + smoothV;
            newObjectiveScore = newObjectiveScore + loss;
            %disp(sprintf('\t\tdomain #%d => empTerm:%f, smoothU:%f, smoothV:%f ==> objective score:%f', i, normEmp, smoothU, smoothV, loss));
        end
        %disp(sprintf('\tEmperical Error:%f', newObjectiveScore));
        %fprintf('iter:%d, error = %f\n', iter, newObjectiveScore);
        diff = oldObjectiveScore - newObjectiveScore;
    end
    
    if newObjectiveScore < bestObjectiveScore
        bestObjectiveScore = newObjectiveScore;
        bestU = U;
    end
    
end

save(sprintf('%sU_%f_%f.mat', directoryName, lambda,gama), 'bestU');

showExperimentInfo(exp_title, datasetId, prefix, numSampleInstance, numSampleFeature, numInstanceCluster, numFeatureCluster, sigma);
fprintf('done\n');