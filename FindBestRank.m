function [ bestRank ] = FindBestRank( x, maxRank )
    sz = size(x);
    %[dx, dy, dz] = size(x)
    dx = sz(1);
    dy = sz(2);
    dz = sz(3);
    tensorX = tensor(x);
    minError = Inf;
    
    for i = 1:(maxRank/5)
        R = 5*i;
        time = round(clock);
        fprintf('Time: %d/%d/%d,%d:%d:%d\n', time(1), time(2), time(3), time(4), time(5), time(6));
        fprintf('Start cp_rank = %d\n', R);
        P = cp_apr(tensorX, R, 'printitn', 0, 'alg', 'mu');%parafac_als(tensorX, R);
        [warnmsg, msgid] = lastwarn;
        if strcmp(msgid,'MATLAB:nearlySingularMatrix')
            lastwarn('');
            break;
        end
        error = 0;
        for iz = 1:dz
            original = x(:,:,iz);
            appx = P.U{1}*diag(P.lambda(:).*P.U{3}(iz,:)')*P.U{2}';
            error = error + abs(original - appx);
        end
        totalError = norm(error)*norm(error);
        if totalError < minError
            minError = totalError;
            bestRank = R;
        end
        time = round(clock);
        fprintf('Time: %d/%d/%d,%d:%d:%d\n', time(1), time(2), time(3), time(4), time(5), time(6));
        fprintf('Finish cp_rank = %d, error = %f\n\n', R, totalError);
    end
    fprintf('Best rank: %d\n', bestRank);
    fprintf('Error: %f\n', minError);
end

