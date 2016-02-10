% Add a sufficiently samll value on inverse if it's ill-conditioned.
% Note: inverse must be squared-matrix.
function [ inverseResult ] = saveRightInverse(M, inverse)
    [dim, ~] = size(inverse);
    smallValue = 0.00000001;
    maxIter = 8;
    iter = 0;
    while(1)
       iter = iter + 1;
       inverseResult = M/inverse;
       [~, warnType] = lastwarn;
       state1 = iter > maxIter;
       state2 = ~strcmp(warnType, 'MATLAB:singularMatrix');
       state3 = ~strcmp(warnType, 'MATLAB:nearlySingularMatrix');
       state4 = all(all(isnan(inverseResult))) == 0;
       state5 = all(all(isinf(inverseResult))) == 0;
       fprintf('Add diag: %d\n', iter);
       if(state1 || state2 && state3 && state4 && state5)
            break;
       else
%            disp('matrix is singular or nearly singular. Add diag.')
           inverse = inverse + smallValue* diag(ones(dim, 1)); 
       end
       smallValue = smallValue* 10;
       lastwarn('');
    end
end

