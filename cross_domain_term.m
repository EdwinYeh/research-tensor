function [Num,Den]=cross_domain_term(Y,W,U,V,CP1,CP2,CP3,CP4,dom,fold,derive,delta)
% CP1=CP1{fold} CP1 is matrix
Num=[];
Den=[];
if dom==1
    A=CP1{fold};B=CP2{fold};C=CP3{fold};D=CP4{fold};
else
    A=CP3{fold};B=CP4{fold};C=CP1{fold};D=CP2{fold};
end
[rC,cC]=size(C);
[rD,cD]=size(D);
Y=Y{dom}; U=U{fold,dom}; V=V{fold,dom}; psi=buildPsi(kr(C,D));
if derive== 'A'
    oneTerm=ones(rC,cC);
    calTerm=diag(sum(D));
else
    oneTerm=ones(rD,cD);
    calTerm=diag(sum(C));
end
save cross_environ
if dom==1
Num=oneTerm*diag(diag((B'*V'*(Y)'*U*A*calTerm)));
Den=delta*oneTerm*(diag(sum(A)).*diag(sum(B)).*calTerm)+oneTerm*diag(diag(B'*V'*((V*B*psi'*A'*U'))*U*A*calTerm));
else
Num=oneTerm*diag(diag((B'*V'*(Y.*W)'*U*A*calTerm)));
Den=delta*oneTerm*(diag(sum(A)).*diag(sum(B)).*calTerm)+oneTerm*diag(diag(B'*V'*((V*B*psi'*A'*U').*W')*U*A*calTerm));
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

