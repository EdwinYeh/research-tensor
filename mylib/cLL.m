LL = zeros(500,4);
for s = 1:500
    LL(s,:) = ones(4,1) * pow(Lu{1}(s, s),0.5);
    
end