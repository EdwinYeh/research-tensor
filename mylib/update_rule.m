function Y=update_rule(Q,A,B,L,W,isU,X)
% Update:U  Q=Y   A=eye(#rows of U) B=V*projB'              L=L  W=W    isU=1  
% Update:V  Q=Y   A=U*projB         B=eye(#columns of V')   L=L  W=W    isU=0
% Update:A  Q=Y   A=U               B=V*E*psi'              L=L  W=W    isU=0
% Update:E  Q=Y   A=U*A*psi         B=V                     L=L  W=W    isU=0
% Output X is a column vector with length = r
    [rx, cx] = size(X);
    [~,c]=size(kron(B,A));
    [~,n]=size(B);
    if isU
        tmpX=((repmat(W(:),[1,c]).*kron(B,A))'*(repmat(W(:),[1,c]).*kron(B,A))+kron((eye(n)),L))\(((repmat(W(:),[1,c])).*kron(B,A))'*(Q(:).*W(:)));
    else
        tmpX=((repmat(W(:),[1,c]).*kron(B,A))'*(repmat(W(:),[1,c]).*kron(B,A)))\((repmat(W(:),[1,c]).*kron(B,A))'*(Q(:).*W(:)));
    end
    Y = reshape(tmpX, rx, cx);
end