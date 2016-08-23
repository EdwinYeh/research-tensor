function [K] = gaussian_kernel(X, sigma)
% X: n*d
    [n,~] = size(X);
    K = zeros(n,n);
    for i=1:n
        for j=i:n
            K(i,j) = exp(-(norm(X(i,:)-X(j,:),2)^2)/(sigma^2));
            K(j,i) = K(i,j);
        end
    end
end