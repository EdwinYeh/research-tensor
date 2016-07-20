function label = predict(input,x,domIdx)
%   input.Tensor = Tensor
%   input.W = W
%   hyperparam.domIdx = domIdx where x belongs to
%   x = predict item
label = x * input.W{domIdx} * projection(input.Tensor,domIdx);


function psi = getPsi(Tensor,domainIdx)
[M,n]=Khatrirao(Tensor,domainIdx);
cpRank=size(Tensor.E{1},2);
psi=zeros(cpRank);
for i = 1:n
    psi = psi + diag(M(i,:));
end

function [M,ProductionOfLabelNum]=Khatrirao(Tensor,domainIdx)
% Khatirao product on all the E matrices in Tensor, except E{domainIdx}
%  Todo : optimize
%           1.the preparation for large Label set (Total #Label cross domain) overflow
%           2.or sparse matrix calculation
%
%%
% Prepare the matrices
cpRank=size(Tensor.E{1},2);
domainNum=length(Tensor.E);
ProductionOfLabelNum = 1;
for DomIdx = 1:domainNum
    %     ignore E{domainIdx}
    if DomIdx == domainIdx
        continue;
    end
    ProductionOfLabelNum = ProductionOfLabelNum * size(Tensor.E{DomIdx},1);
end
M=sparse( ProductionOfLabelNum ,cpRank);


%%
for r = 1:cpRank
    M(:,r);
    E_col=1;
    % Note that E_col is a column
    for  DomIdx = 1:domainNum
        %     ignore E{domainIdx}
        if DomIdx == domainIdx
            continue;
        end
        tmpMatrix = Tensor.E{DomIdx};
        E_col=kron(E_col,tmpMatrix(:,r));
    end
    M(:,r)=M(:,r)+E_col;
end

function proj = projection(Tensor,domainIdx)
psi = getPsi(Tensor,domainIdx);
proj = Tensor.A * psi * Tensor.E{domainIdx}';
