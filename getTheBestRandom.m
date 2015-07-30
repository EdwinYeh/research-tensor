function [U, B, V] = getTheBestRandom(X, W, Y, numInstance, numFeature, numInstanceCluster, numFeatureCluster, label, validateIndex, numDom, targetDomain, tryTime, isMotar)

V = cell(1, numDom);
U = cell(1, numDom);
bestV = cell(1, numDom);
bestU = cell(1, numDom);
loss = cell(1, numDom);
bestLoss = 999999999;

str = '';
for i = 1:numDom
    str = sprintf('%s%d,%d,', str, numInstanceCluster(i), numFeatureCluster(i));
end
str = str(1:length(str)-1);
%random initialize B
randStr = eval(sprintf('rand(%s)', str), sprintf('[%s]', str));
randStr = round(randStr);

for t = 1:tryTime
    if isMotar == true
        B = tensor(randStr);
    else
        B = rand(numInstanceCluster(1), numFeatureCluster(1));
    end
    parfor i = 1:numDom
        V{i} = rand(numFeature(i),numFeatureCluster(i));
        U{i} = rand(numInstance(i), numInstanceCluster(i));
        
        if i == targetDomain
            U{i} = fixTrainingSet(U{i}, label{i}, validateIndex);
        else
            U{i} = Y{i};
        end
        
        if isMotar == true
            U{i}(:, numInstanceCluster(i)) = 0;
            V{i}(:, numFeatureCluster(i)) = 0;
            [projB, ~] = SumOfMatricize(B, 2*(i - 1)+1);
            result = U{i}*projB*V{i}';
        else
            result = U{i}*B*V{i}';
        end
        normEmp = norm(W{i}.*(X{i} - result))*norm(W{i}.*(X{i} - result));
        loss{i} = normEmp;
    end
    totalLoss = 0;
    for i = 1:numDom
        totalLoss = totalLoss + loss{i};
    end
    if totalLoss < bestLoss
        bestU = U;
        bestV = V;
        bestB = B;
    end
end
U = bestU;
B = bestB;
V = bestV;
end