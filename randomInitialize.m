function [U, B, V] = randomInitialize(numInstance, numFeature, numInstanceCluster, numFeatureCluster, numDom, isMotar)

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

    for i = 1:numDom
        V{i} = rand(numFeature(i),numFeatureCluster(i));
        U{i} = rand(numInstance(i), numInstanceCluster(i));
    end
end