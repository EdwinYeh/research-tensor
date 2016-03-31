function [A,psi,E]=projectTensorToMatrix(CP_matrices,DomId)
% It's only fit for 4-way tensor
% CP_matrices is cell of {A,B,C,D}
% DomId=1 if source domain, 2 if target domain

    if DomId ==1 
        A=CP_matrices{1};
        E=CP_matrices{2};
        psi=buildPsi(kr(CP_matrices{3},CP_matrices{4}));
    else
        A=CP_matrices{3};
        E=CP_matrices{4};
        psi=buildPsi(kr(CP_matrices{1},CP_matrices{2}));
    end
    
    function M =kr(A,B)
            [Ar,r]=size(A);
            [Br,r]=size(B);
            M=zeros(Ar*Br,r);
            for  i=1:r
                M(:,i)=kron(A(:,i),B(:,i));
            end
    end

    function psi=buildPsi(M)
        [c,r]=size(M);
        psi=zeros(r);
        for i=1:c
            psi=psi+diag(M(i,:));
        end
    end
end