function tmpObjectiveScore = showObjectiveDX(U, V, X, Lu, Lv, CP1, CP2, CP3, CP4, lambda, gama)
    tmpObjectiveScore = 0;
    for dom = 1:2
        %for i = 1:numDom
        [A,sumFi,E] = projectTensorToMatrix({CP1,CP2,CP3,CP4}, dom);
        projB = A*sumFi*E';
        result = U{dom}*projB*V{dom}';
        normEmp = norm((X{dom} - result), 'fro')*norm((X{dom} - result), 'fro');
        smoothU = lambda*trace(U{dom}'*Lu{dom}*U{dom});
        smoothV = gama*trace(V{dom}'*Lv{dom}*V{dom});
        objectiveScore = normEmp + smoothU + smoothV;
        tmpObjectiveScore = tmpObjectiveScore + objectiveScore;
%         fprintf('\t\tdomain #%d => empTerm:%g, smoothU:%g, smoothV:%g ==> objective score:%g\n', dom, normEmp, smoothU, smoothV, objectiveScore);
    end
end