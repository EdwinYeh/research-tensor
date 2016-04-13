function obj=objective(CP,Y,U,V,Lu,W,lambda,delta,numDom,targetDomain,normType)
	newObjectiveScore=0;
	for domId = 1:numDom
		[A,sumFi,E]=projectTensorToMatrix(CP,domId);
		projB=A*sumFi*E';
		result = U{domId}*projB*V{domId}';
		if domId == targetDomain
			normEmp = norm((Y{domId} - result).*W,'fro')^2;
		else
			normEmp = norm((Y{domId} - result),'fro')^2;
		end
		smoothU = lambda*trace(U{domId}'*Lu{domId}*U{domId});
		normH=delta*norm(projB,normType);
		objectiveScore = normEmp + smoothU + normH;
		newObjectiveScore= newObjectiveScore + objectiveScore;
	end
	obj=newObjectiveScore;
	
end
