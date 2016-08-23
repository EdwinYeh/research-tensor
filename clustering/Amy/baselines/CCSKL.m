function [ Final ] = CCSKL( X, k, mustLink, cannotLink ,m)
%CCSKL Summary of this function goes here
%   Detailed explanation goes here
	[~,n] = size(X);
    
    sigma = zeros(n,1);
    PD = squareform(pdist(X'));
    for i = 1:n
        [~,idx] = sort(PD(i,:));
        sigma(i) = PD(i,idx(10));
    end
    
    W = zeros(n,n);

    for i=1:n
       for j=1:n
           W(i,j) = exp(-norm(X(:,i)-X(:,j))^(2)/(sigma(i)*sigma(j)));
       end
    end
    D = diag(sum(W));
    I = diag(ones(n,1));

    L = I-D^(-1/2)*W*D^(-1/2);
    [evec,eval] = eig( L);
    [~,idx] = sort(diag(eval));
    evec = evec(:,idx);
    F = evec(:,1:m);


% calculate C
    C = zeros(n,n);
    for i=1:size(mustLink,1)
        idx = find(mustLink(i,:)~=0);
        C(idx(1),idx(2)) = 1;
        C(idx(2),idx(1)) = 1;
    end
    for i=1:size(cannotLink,1)
        idx = find(cannotLink(i,:)~=0);
        C(idx(1),idx(2)) = 1;
        C(idx(2),idx(1)) = 1;
    end
    C = C+diag(ones(n,1));
    
    A = zeros(m,m);
    for i=1:n
       for j=1:n
           if C(i,j)~=0
               yij = F(i,:).*F(j,:);
               A = A+ C(i,j)*yij'*yij;
           end
       end
    end
    A = A*2;
    
    b = zeros(m,1);
     for i=1:n
       if C(i,j)~=0
           yij = F(i,:).*F(j,:);
           b = b+ C(i,j)*yij';
       end
    end
    
    cvx_begin quiet
        variable z(m)
        minimize (1/2*z'*A*z+b'*z)
        subject to
            z >= [z(2:m);0]
    cvx_end
    
    Lambda = diag(z);
    Final = 100*F* (Lambda^(1/2));


end

