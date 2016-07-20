load environmentForNewModel
output = solver(input,hyperparam);

for domID = 1:numDom
    correctCount = 0;
    for testDataIndex = 1: numTestData(domID)
        hyperparam.domIdx = domID;
        [~, predictResult] = max(predict(output, XTest{domID}(testDataIndex, :), domID));
        if LabelTest{domID}(testDataIndex) == predictResult
            correctCount = correctCount + 1;
        end
    end
    fprintf('domain: %d, accuracy: %g\n', domID, correctCount/numTestData(domID))
end
