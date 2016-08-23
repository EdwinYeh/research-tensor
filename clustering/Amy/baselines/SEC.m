function [F] = SEC(X, r, mu, gamma)
% X\in d*n
    %parameter

    [d,n] = size(X);
    
    % X*1n = 0, means the amount of each row should be 0
    for i=1:d
        offset = mean(X(i,:))/n;
        X(i,:) = X(i,:) - offset;
    end
    
    sigma = zeros(n,1);
    PD = squareform(pdist(X'));
    for i = 1:n
        [~,idx] = sort(PD(i,:));
        sigma(i) = PD(i,idx(7));
    end
    clear PD;
    
    A = zeros(n,n);

    for i=1:n
       for j=1:n
           A(i,j) = exp(-norm(X(:,i)-X(:,j))^(2)/(sigma(i)*sigma(j)));
       end
    end
    clear sigma;
    D = diag(sum(A));
    I = diag(ones(n,1));

    L = I-D^(-1/2)*A*D^(-1/2);
    Hc = I-(1/n)*ones(n,1)*ones(1,n);
    clear I D A;
    tmp = L + mu*gamma*Hc-mu*gamma*gamma*X'/(gamma*X*X'+diag(ones(d,1)))*X;
    [evec,eval] = eig(tmp);
    [~,idx] = sort(diag(eval));
    clear eval;
    evec = evec(:,idx);

    F = evec(:,1:r);

    %the columns of F* are from the bottom c eigenvedctors
end