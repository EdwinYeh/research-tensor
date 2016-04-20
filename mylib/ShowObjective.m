function tmpObjectiveScore = ShowObjective(fold, U, V, W, YMatrix, Lu, CP1, CP2, CP3, CP4, lambda)
    tmpObjectiveScore = 0;
    for dom = 1:2
        [A,sumFi,E] = projectTensorToMatrix({CP1{fold},CP2{fold},CP3{fold},CP4{fold}}, dom);
        projB = A*sumFi*E';
        result = U{fold, dom}*projB*V{fold, dom}';
        if dom == 2
            normEmp = norm((YMatrix{dom} - result).*W, 'fro')*norm((YMatrix{dom} - result).*W, 'fro');
        else
            normEmp = norm((YMatrix{dom} - result), 'fro')*norm((YMatrix{dom} - result), 'fro');
        end
        smoothU = lambda*trace(U{fold, dom}'*Lu{dom}*U{fold, dom});
        objectiveScore = normEmp + smoothU;
        tmpObjectiveScore = tmpObjectiveScore + objectiveScore;
    end
%     disp(tmpObjectiveScore);
end