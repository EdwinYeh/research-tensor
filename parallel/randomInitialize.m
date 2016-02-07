function [U, B, V] = randomInitialize(numInstance, numFeature, numInstanceCluster, numFeatureCluster, numDom, isMotar)

    V = cell(1, numDom);
    U = cell(1, numDom);

    str = '';
    for i = 1:numDom
        str = sprintf('%s%d,%d,', str, numInstanceCluster, numFeatureCluster);
    end
    str = str(1:length(str)-1);
    randStr = eval(sprintf('rand(%s)', str), sprintf('[%s]', str));
    randStr = round(randStr);

    if isMotar == true
        B = tensor(randStr);
    else
        rng('shuffle');
        B = rand(numInstanceCluster, numFeatureCluster);
    end

    for i = 1:numDom
        rng('shuffle');
        V{i} = rand(numFeature(i),numFeatureCluster);
        U{i} = rand(numInstance(i), numInstanceCluster);
    end
end