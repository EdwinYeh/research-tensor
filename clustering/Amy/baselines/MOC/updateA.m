function newA = updateA(X,M,A)
    newA = pinv(M)*X;
end