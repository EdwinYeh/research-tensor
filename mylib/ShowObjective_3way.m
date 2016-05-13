function tmpObjectiveScore = ShowObjective_3way(fold, U, V, W, YMatrix, Lu, CP1, CP2, CP3, lambda, delta)
    tmpObjectiveScore = 0;
	A=CP1{fold};
    for dom = 1:2
%        [A,sumFi,E] = projectTensorToMatrix({CP1{fold},CP2{fold},CP3{fold},CP4{fold}}, dom);
	if dom==1
	M=CP3{fold};
	E=CP2{fold};
	else
	M=CP2{fold};
	E=CP3{fold};
	end
	sumFi=buildPsi(M);
        projB = A*sumFi*E';
        result = U{fold,dom}*projB*V{fold, dom}';
        if dom == 2
            normEmp = norm((YMatrix{dom} - result).*W, 'fro')*norm((YMatrix{dom} - result).*W, 'fro');
        else
            normEmp = norm((YMatrix{dom} - result), 'fro')*norm((YMatrix{dom} - result), 'fro');
        end
        smoothU = lambda*trace(U{fold,dom}'*Lu{dom}*U{fold,dom});
        oneNormH = delta*norm(projB, 1);
        objectiveScore = normEmp + smoothU + oneNormH;
        tmpObjectiveScore = tmpObjectiveScore + objectiveScore;
    end
%     disp(tmpObjectiveScore);


function psi=buildPsi(M)
[c,r]=size(M);
psi=zeros(r);
for i=1:c
    psi=psi+diag(M(i,:));
end

