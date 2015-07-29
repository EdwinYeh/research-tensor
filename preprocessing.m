function [g s idx] = preprocessing(numDom, predDomain)
    g = 0;
    s = 0;
    idx = 0;
    for i = 1:numDom
        oriX = importdata(sprintf('RealData/realX%d.mat', i));
        %oriX = importdata(sprintf('Three5X%d.mat', i));
        sizeX = size(oriX);
        sizeX
        snapshot = zeros(sizeX(1), sizeX(2));
        
        if i == predDomain
            tmp = sum(oriX, 3);
            
            %nonzeroidx = find(tmp);
            %idx = nonzeroidx(ceil(length(nonzeroidx)*rand(1,10)));
            idx = 0;
            
            snapshot = tmp - oriX(:,:,sizeX(3));
            %snapshot(idx) = snapshot(idx) - tmp(idx);
            groundTruthX = tmp;
            g = groundTruthX;
            s = snapshot;
        else
            snapshot = sum(oriX, 3);
        end
        save(sprintf('RealData/realX%dFinal.mat', i), 'snapshot');
        
    end
end
