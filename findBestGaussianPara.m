datasetId = 1;
sigma = 0.001;
searchSpeed = 10;
tryTime = 6;
source = load(sprintf('../20-newsgroup/source%d.csv', datasetId));
target = load(sprintf('../20-newsgroup/target%d.csv', datasetId));

source = sparse(source(:, 1), source(:, 2), source(:, 3));
target = sparse(target(:, 1), target(:, 2), target(:, 3));

source = source(:, findDenseFeature(source, 20));
target = target(:, findDenseFeature(target, 20));

[numSourceInstance, ~] = size(source);
[numTargetInstance, ~] = size(target);

source(randperm(numSourceInstance, 1000), :);
target(randperm(numTargetInstance, 1000), :);
sourceSVector = zeros((1 + numSourceInstance)*numSourceInstance/2);
targetSVector = zeros((1 + numTargetInstance)*numTargetInstance/2);
sourceSMatrix = zeros(numSourceInstance, numSourceInstance);
targetSMatrix = zeros(numTargetInstance, numTargetInstance);

for t = 1:tryTime
    disp(sigma);
    index = 1;
    for i = 1:numSourceInstance
        for j = 1:numSourceInstance
            if j >= i
                dif = norm(source(i, :) - source(j, :));
                gaussianSimilarity = exp(-(dif*dif)/(2*sigma));
                sourceSVector(index) = gaussianSimilarity;
                sourceSMatrix(i, j) = gaussianSimilarity;
                sourceSMatrix(j, i) = gaussianSimilarity;
                index = index + 1;
            end
        end
    end
    fprintf('source median = %f\n', median(sourceSVector));
    index = 1;
    for i = 1:numTargetInstance
        for j = 1:numTargetInstance
            if j >= i
                dif = norm(target(i, :) - target(j, :));
                gaussianSimilarity = exp(-(dif*dif)/(2*sigma));
                targetSVector(index) = gaussianSimilarity;
                targetSMatrix(i, j) = gaussianSimilarity;
                targetSMatrix(j, i) = gaussianSimilarity;
                index = index + 1;
            end
        end
    end
    fprintf('target median = %f\n', median(targetSVector));
    sigma = sigma * searchSpeed;
end