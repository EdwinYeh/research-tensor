function [ output ] = normr( input )
    [row, col] = size(input);
    output = zeros(row, col);
    for r = 1:row
        output(r, :) = input(r, :)/ norm(input(r, :)); 
    end
end

