function [ projB, threeMtrixB ] = SumOfMatricize( tensorB, mode )
    tensorSize = tensorB.size;
    rdim = tensorSize(mode);
    cdim = tensorSize((mode + 1));
    matricizeB = double(tenmat(tensorB, mode, 'fc'));
    Mcell = mat2cell(matricizeB, rdim, repmat(cdim,length(matricizeB)/cdim,1));
    threeMtrixB = zeros(rdim, cdim, length(Mcell));
    divMatrix = zeros(rdim, cdim, length(Mcell));
    for i = 1:length(Mcell)
        threeMtrixB(:,:,i) = Mcell{i};
    end
    divMatrix(threeMtrixB~=0) = 1;
    projB = sum(threeMtrixB, 3);%./sum(divMatrix, 3);%sum(threeMtrixB, 3);
    projB(isnan(projB)) = 0;
    %threeTensorB = tensor(threeTensorB);
end

