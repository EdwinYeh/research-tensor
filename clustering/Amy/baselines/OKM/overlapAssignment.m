function [ A ] = overlapAssignment ( X, means, cannotLink, Atmp, Aold )
%ASSIGNOVERLAP Summary of this function goes here
%   Detailed explanation goes here
    [c,~] = size(means);
    [Ndata, ~] = size(X);
   
    A = Atmp;
    Anew = A;
   
    
    for i=1:Ndata
        phi = mean(means(logical(A(i,:)),:),1);
        x = X(i,:);
        for j=1:c
            % Euclidean distance from data to each prototype
            dist(j) = norm(x-means(j,:))^2;
        end
         
        % step2
        while 1
            if sum (A(i,:))==c
                break;
            end
            
            
             % cannot link constraint
            avaCluster = ones(c,1);
            idx = find(cannotLink(:,i)~=0);
            if ~isempty(idx)
                link =  find(sum(abs(cannotLink(idx,:)),1)>0);
                for z=1:length(link);
                    avaCluster(A(link(z),:)==1)=0;
                end
            end
            
            distance = dist((logical(avaCluster')&(~A(i,:))));
            % Find indices of minimum distance
            if isempty(distance)
                break;
            end
            index_min = find(~(dist-min(distance)));

            tmpidx = [];
            for k=1:length(index_min)
                if A(i,index_min(k))==1
                    tmpidx = [tmpidx k];
                end
            end
            for k=length(tmpidx):-1:1
                index_min = index_min(:,[1:tmpidx(k)-1, tmpidx(k)+1:length(index_min)]);
            end


            % If there are multiple min distances, decide randomly
            index_min = index_min(ceil(length(index_min)*rand));
            Anew(i, index_min) = 1;



            phinew = mean(means(logical(Anew(i,:)),:),1);

            if norm(x-phinew) <= norm(x-phi) || nargin ==4
                A(i,:) = Anew(i,:);
                phi = phinew;

            elseif sum(sum(Aold(i,:)))==0
                break;
            else          
                phiold = mean(means((logical(Aold(i,:))&avaCluster'),:),1);

                if norm(x-phiold) < norm(x-phi)
                    A(i,:) = Aold(i,:)&avaCluster';
                end

                break;
            end
        end


    end
end

