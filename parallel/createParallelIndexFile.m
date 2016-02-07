function createParallelIndexFile( datasetId, numParallelPartition )
    indexIntervalStart = 1;
    indexIntervalEnd = 500;
    index = indexIntervalStart:indexIntervalEnd;
    parallelId =1;
    
    for i = 1: numParallelPartition
        csvwrite(sprintf('sampleIndex/sampleSourceDataIndex%d_%d.csv', datasetId, parallelId), index);
        csvwrite(sprintf('sampleIndex/sampleTargetDataIndex%d_%d.csv', datasetId, parallelId), index);
        parallelId = parallelId + 1;
        index = index + 400;
    end
end

