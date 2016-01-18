function [ output_matrix ] = normr( input_matrix )
    sizeOfMatrix = size(input_matrix);
    numOfRow = sizeOfMatrix(1);
    for row = 1:numOfRow
        input_matrix(row, :) = input_matrix(row, :)/ norm(input_matrix(row, :));
    end
    output_matrix = input_matrix;
end

