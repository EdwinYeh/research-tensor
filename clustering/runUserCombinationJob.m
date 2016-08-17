function runUserCombinationJob( minUserId, maxUserId, minJobId, maxJobId )
    jobCombination = combinator(maxUserId,minUserId,'c');
    for jobId = minJobId:maxJobId
        fprintf('Run job:[%d,%d]\n', jobCombination(jobId,1), jobCombination(jobId,2));
        Exp_mturk(jobCombination(jobId,:));
    end
end

