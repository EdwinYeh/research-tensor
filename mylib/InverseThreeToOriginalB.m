function [ originalB ] = InverseThreeToOriginalB( threeB, mode, originalSize )
    matricizeB = double(tenmat(threeB, 1, 'fc'));
    seq = [(mode + 1):length(originalSize), 1:(mode - 1)];
    originalB = tensor(tenmat(matricizeB, mode, seq, originalSize));
end

