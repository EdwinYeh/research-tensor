T = tensor(rand(5,5,5,5));
cpRank = 20;
for cpRank = 5:5:30
    disp(cpRank);
    for projectDomain = 1:2
        CP = cp_apr(T, cpRank, 'printitn', 0, 'alg', 'mu');
        A = CP.U{1};
        B = CP.U{2};
        C = CP.U{3};
        D = CP.U{4};
        CPLambda = CP.lambda(:);
        [A1, spi1, E1] = projection({A,B,C,D}, projectDomain, CPLambda);
        
        [projB, threeMatrixB] = SumOfMatricize(T, 2*(projectDomain - 1)+1);
        CP = cp_apr(tensor(threeMatrixB), cpRank, 'printitn', 0, 'alg', 'mu');%parafac_als(tensor(threeMatrixB), bestCPR);
        A2 = CP.U{1};
        E2 = CP.U{2};
        C = CP.U{3};
        CPLamda = CP.lambda(:);
        fi = cell(1, length(CP.U{3}));
        [r, c] = size(C);
        spi2 = zeros(c, c);
        for idx = 1:r
            fi{idx} = diag(CPLamda.*C(idx,:)');
            spi2 = spi2 + fi{idx};
        end
        disp(norm(spi1-spi2, 'fro'));
    end
end