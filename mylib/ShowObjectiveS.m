function tmpObjectiveScore = ShowObjectiveS(SU,SV,U, V, W, YMatrix, Lu, CP1, CP2, CP3, CP4, lambda)
    tmpObjectiveScore = 0;
    for i = 1:2
        [A,sumFi,E] = projectTensorToMatrix({CP1,CP2,CP3,CP4}, i);
        projB = A*sumFi*E';
        result = SU{i}*U{i}*projB*V{i}'*SV{i}';
        if i == 2
            normEmp = norm((YMatrix{i} - result).*W, 'fro')*norm((YMatrix{i} - result).*W, 'fro');
        else
            normEmp = norm((YMatrix{i} - result), 'fro')*norm((YMatrix{i} - result), 'fro');
        end
        smoothU = lambda*trace(U{i}'*Lu{i}*U{i});
        objectiveScore =normEmp + smoothU;
        tmpObjectiveScore = tmpObjectiveScore + objectiveScore;
    end
%     disp(tmpObjectiveScore);
end
