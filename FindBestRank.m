function [ bestRank ] = FindBestRank( x, maxRank )
    sz = size(x);
    %[dx, dy, dz] = size(x)
    dx = sz(1);
    dy = sz(2);
    dz = sz(3);
    tensorX = tensor(x);
    totalErrors = Inf(1, maxRank);
    for R = 1:maxRank
        P = cp_apr(tensorX, R, 'printitn', 0);%parafac_als(tensorX, R);
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
        totalErrors(R) = totalError;
    end
    totalErrors(isnan(totalErrors)) = Inf;
    bestRank = find(totalErrors==min(totalErrors(1:R)));
    bestRank = bestRank(1);
end

