function [Num,Den]=cross_domain_term_3way(Y,W,U,V,CP1,CP2,CP3,dom,fold,delta)
% CP1=CP1{fold} CP1 is matrix
Num=[];
Den=[];
if dom==1
    A=CP1{fold};B=CP2{fold};C=CP3{fold};
else
    A=CP1{fold};B=CP3{fold};C=CP2{fold};
end
[rC,cC]=size(C);
Y=Y{dom}; U=U{fold,dom}; V=V{fold,dom}; psi=buildPsi(C);
% if derive== 'A'
     oneTerm=ones(rC,cC);
     %calTerm=diag(sum(D));
% else
%     oneTerm=ones(rD,cD);
%     calTerm=diag(sum(C));
% end
%save cross_environ
if dom==1
Num=oneTerm*diag(diag((B'*V'*(Y)'*U*A)));
Den=delta*oneTerm*(diag(sum(A)).*diag(sum(B)))+oneTerm*diag(diag(B'*V'*((V*B*psi'*A'*U'))*U*A));
else
Num=oneTerm*diag(diag((B'*V'*(Y.*W)'*U*A)));
Den=delta*oneTerm*(diag(sum(A)).*diag(sum(B)))+oneTerm*diag(diag(B'*V'*((V*B*psi'*A'*U').*W')*U*A));
end

function M =kr(A,B)
[Ar,r]=size(A);
[Br,r]=size(B);
M=zeros(Ar*Br,r);
for  i=1:r
    M(:,i)=kron(A(:,i),B(:,i));
end

function psi=buildPsi(M)
[c,r]=size(M);
psi=zeros(r);
for i=1:c
    psi=psi+diag(M(i,:));
end

