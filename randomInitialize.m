function [U, B, V] = randomInitialize(Y, numInstance, numFeature, numInstanceCluster, numFeatureCluster, label, validateIndex, numDom, targetDomain, isMotar)

V = cell(1, numDom);
U = cell(1, numDom);

str = '';
for i = 1:numDom
    str = sprintf('%s%d,%d,', str, numInstanceCluster(i), numFeatureCluster(i));
end
str = str(1:length(str)-1);
randStr = eval(sprintf('rand(%s)', str), sprintf('[%s]', str));
randStr = round(randStr);

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
end
end