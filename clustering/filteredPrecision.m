function [score] = filteredPrecision(trueC, foundC)
    [~,k] = size(foundC);
    validInstances = find(sum(trueC,2));
    trueC = trueC(validInstances,:);
    foundC = foundC(validInstances,:);
    
    %calculate precision
    score = 0;
    for i=1:k
        base = sum(foundC(:,i));
        if(base == 0)
            continue;
        end
        overlap = sum(trueC(:,i) & foundC(:,i));
        score = score + (overlap/base);
    end
    score = score*(1/k);
    
end