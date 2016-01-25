function M=constructM(B,C)
        [n,~]=size(B);
        [n2,~] = size(C);
        M=blkdiag(C);
        for i=1:n-1
            M=blkdiag(C,M);
        end
        M=M+kron(B,ones(n2));
end