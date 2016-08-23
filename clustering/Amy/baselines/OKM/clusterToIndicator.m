function [ indicator ] = clusterToIndicator( cluster )
    c = max(cluster);
    n = length(cluster);
    indicator = zeros(c, n);

    for i=1:c
       indicator(i, cluster==i )=1;
    end

end

