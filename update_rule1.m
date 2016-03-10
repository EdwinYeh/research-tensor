function X=update_rule(Q,A,B,L,W,isU,isA,projB,lambda)
% Update:U  Q=Y   A=eye(#rows of U) B=V*projB'              L=L  W=W    isU=1	isA=0
% Update:V  Q=Y   A=U*projB         B=eye(#columns of V')   L=L  W=W    isU=0   isA=0
% Update:A  Q=Y   A=U               B=V*E*psi'              L=L  W=W    isU=0	isA=1
% Update:E  Q=Y   A=U*A*psi         B=V                     L=L  W=W    isU=0	isA=0


    [r,c]=size(kron(B,A));
    [m,n]=size(B);
	[k,~]=size(projB);
	[nI,nT]=size(A);
	M=(repmat(W(:),[1,c]).*kron(B,A));
	ATA=M'*M;
	AAT=M*M';
    if isU
		if rcond(ATA)==0
			%X=pinv(ATA+lambda*kron((eye(n)),L))*(M'*(Q(:).*W(:)));
			X=M'*pinv(AAT+lambda*kron((eye(m)),L))*(Q(:).*W(:));
		else
			X=(ATA+lambda*kron((eye(n)),L))\(M'*(Q(:).*W(:)));
		end
		X=reshape(X,[nI,k]);
    else
		if rcond(ATA)==0
			X=pinv(M)*(Q(:).*W(:));
		else
			X=(ATA)\(M'*(Q(:).*W(:)));
		end
		if isA
			X=reshape(X,[k,n]);
		else
			X=reshape(X,[n,nT]);
    end
    
end