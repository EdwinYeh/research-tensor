function [ A ] = assignOverlap( x, means, seeds, Aold )
%ASSIGNOVERLAP Summary of this function goes here
%   Detailed explanation goes here
    [c,d] = size(means);
    %step 1
    for j=1:c
        % Euclidean distance from data to each prototype
        dist(j) = norm(x-means(j,:))^2;
    end

    if sum(seeds)==0    
        % Find indices of minimum distance
        index_min = find(~(dist-min(dist)));
        % If there are multiple min distances, decide randomly
        index_min = index_min(ceil(length(index_min)*rand));
        A = zeros(1,c);
        A(1, index_min) = 1;
    else
        A = seeds;
    end
    
    Anew = A;
    phi = mean(means(logical(A),:),1);
    
    % step2
    while 1
        if sum (A)==c
            break;
        end
        % Find indices of minimum distance
        index_min = find(~(dist-min(dist(~A))));

        tmpidx = [];
        for i=1:length(index_min)
            if A(index_min(i))==1
                tmpidx = [tmpidx i];
            end
        end
        for i=length(tmpidx):-1:1
            index_min = index_min(:,[1:tmpidx(i)-1, tmpidx(i)+1:length(index_min)]);
        end
   

        % If there are multiple min distances, decide randomly
        index_min = index_min(ceil(length(index_min)*rand));
        Anew(1, index_min) = 1;

        
        
        phinew = mean(means(logical(Anew),:),1);

        if norm(x-phinew) <= norm(x-phi)          
            A = Anew;
            phi = phinew;
            
        elseif sum(sum(Aold))==0
            break;
        else          
            phiold = mean(means(logical(Aold),:),1);
            
            if norm(x-phiold) < norm(x-phi)
                A = Aold;
            end
            
            break;
        end
    end
 
    
end

