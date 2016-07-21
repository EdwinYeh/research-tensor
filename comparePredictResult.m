function accuracy = comparePredictResult(YTrue,Y)
    correctCount = 0;
    for i = 1:size(Y, 1)
	[~, labelTrue] = max(YTrue(i,:));
	[~, labelPredict] = max(Y(i,:));
        if isequal(labelTrue, labelPredict)
            correctCount = correctCount + 1;
        end
    end
    accuracy = correctCount/size(Y,1);
end
