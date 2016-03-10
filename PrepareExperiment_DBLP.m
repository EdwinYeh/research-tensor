% sparse form
prefix = '../DBLP/';
% map datsetId 14 to DBLP 1
fileName = sprintf('%sDBLP%d.mat', prefix, (datasetId-13));

TrueYMatrix = cell(1, numDom);
YMatrix = cell(1, numDom);
Label = cell(1, numDom);
Su = cell(1, numDom);
Du = cell(1, numDom);
Lu = cell(1, numDom);

initV = cell(randomTryTime, numDom);
initU = cell(randomTryTime, numDom);
initB = cell(randomTryTime);

sampleSourceDataIndex = csvread(sprintf('sampleIndex/sampleSourceDataIndex%d.csv', datasetId));
sampleTargetDataIndex = csvread(sprintf('sampleIndex/sampleTargetDataIndex%d.csv', datasetId));

load(fileName);

for dom = 1: numDom
    if dom == sourceDomain
        Label{dom} = label1(sampleSourceDataIndex, :);
        Su{dom} = edges1(sampleSourceDataIndex, sampleSourceDataIndex) + 0.001*ones(numSampleInstance(dom), numSampleInstance(dom));
%         Label{dom} = label1(:, :);
%         Su{dom} = edges1(:, :);
    elseif dom ==targetDomain
        Label{dom} = label2(sampleTargetDataIndex, :);
        Su{dom} = edges2(sampleTargetDataIndex, sampleTargetDataIndex) + 0.001*ones(numSampleInstance(dom), numSampleInstance(dom));
%         Label{dom} = label2(:, :);
%         Su{dom} = edges2(:, :);
    end
    [numSampleInstance(dom), ~] = size(Su{dom});
    Du{dom} = zeros(numSampleInstance(dom), numSampleInstance(dom));
    Lu{dom} = zeros(numSampleInstance(dom), numSampleInstance(dom));
    TrueYMatrix{dom} = -1* ones(numSampleInstance(dom), numClass(dom));
    for j = 1: numSampleInstance(dom)        
        TrueYMatrix{dom}(j, Label{dom}(j)) = 1;
    end
    fprintf('Domain%d: calculating Du, Lu\n', dom);
    for useri = 1:numSampleInstance(dom)
        Du{dom}(useri,useri) = sum(Su{dom}(useri,:));
    end
    Lu{dom} = Du{dom} - Su{dom};
end

str = '';
for i = 1:numDom
    str = sprintf('%s%d,%d,', str, numInstanceCluster, numFeatureCluster);
end
str = str(1:length(str)-1);
eval(sprintf('originalSize = [%s];', str));

%Randomly initialize B, U, V
if isRandom == true
    for t = 1: randomTryTime
        [initU(t,:),initB{t},initV(t,:)] = randomInitialize(numSampleInstance, numClass, numInstanceCluster, numFeatureCluster, numDom, isUsingTensor);
    end
end

CVFoldSize = numSampleInstance(targetDomain)/ numCVFold;
clear domain1 domain2 Conf1_ID Conf2_ID edges1 edges2 label1 label2;