function assignTargetU(sourceX, sourceLabel, targetX)
    [sourceLength ,sourceWidth] = size(sourceX);
    sourceXNonDense = zeros(sourceLength, sourceWidth);
    for i = 1:sourceLength
        for j = 1:sourceWidth
            sourceXNonDense(sourceLength, sourceWidth) = sourceX(sourceLength, sourceWidth);
        end
    end
    b = glmfit(sourceXNonDense, sourceLabel, 'normal');
    [targetLength ,targetWidth] = size(targetX);
    targetXNonDense = zeros(targetLength, targetWidth);
    for i = 1:targetLength
        for j = 1:targetWidth
            targetXNonDense(targetLength, targetWidth) = targetX(targetLength, targetWidth);
        end
    end
    targetLabel = glmval(b, targetXNonDense)
end