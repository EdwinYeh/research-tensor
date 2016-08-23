function [F, C, i, objFinal] = solver4(K, P, S, alpha, beta)
 
 % min(F,C>=0) ||KC-F||^2_F + alpha*Tr(C'KC) + beta*||S(F-P)||^2_F
 
 % gamma term is the l1 norm of F's rows, used to ensure the sparsity of cluster
 % assignment(reduce the overlapping degree, increase the precision)
 
 % mu term prevent from getting all 0 cluster assignment for an
 % instance. We can give 1_n a coefficient to control the overlapping rate.
 % We can even give different clusters different coefficients based on their
 % perception similarity to control the overlapping level between clusters (add a term: ||F'F-A||^2_F, where A is a cluster similarity matrix)

    % input - known variables:
        % K \in R^(n*n): data kernel matrix
        % P = perception over instances, n*p (only seeds have values)
        % normalized to ||P_i|| = 1 (the length of each column = 1)
        % S: n*n binary diagonal matrix
    % unknown
        % C \in R^(n*sf) 
        % F = perception feature space, n*p
        
		
	stop = 0.00005;
	maxIter = 10000;	
	
    % initialed outside
    [n, p] = size(P);
   
    % inital value must be nonnegative
    F = rand(n,p);
    C = rand(n,p);
    %etaC = 0.01;
    
    obj(1) = objective(C, F);
    
    i=0;
    
    while 1
        i=i+1;
        %if(mod(i,5000)==0) 
        %    i
        %    obj(i)
        %end
        [gradF] = funF(F);
        Fnew = gradF;
            
        [gradC] = funC(C);
        Cnew = gradC;
        
        obj(i+1)= objective(Cnew, Fnew);
     %  disp( obj(i+1));

        if ((obj(i)-obj(i+1)>=0 && obj(i)-obj(i+1)<stop))
           %figure(16);
           %plot(obj(100:end));
           %saveas(gca,['g' num2str(i)],'bmp');
           break; 
        end
        
		if(i+1>maxIter)
            %figure(16);
            %plot(obj(10:end));
            break;
        end
        
        %if (i+1>20000)
        %   plot(obj(100:end));
        %   saveas(gca,['g' num2str(i)],'bmp');
        %    i=0;
        %    break;
        %end

        C = Cnew;
        F = Fnew;

    end
    objFinal = obj(length(obj));
    
    
    function [obj] = objective(C, F)
        obj = norm(K*C-F,'fro') + alpha*trace(C'*K*C) + beta*norm(S*F-S*P,'fro');
    end
    
    function [grad] = funF(F) 
        grad = F .* sqrt((K*C+beta*S'*S*P)./(F+beta*S'*S*F));
    end

    function [grad] = funC(C)
        grad = C .* sqrt((K*F)./(K*K*C+alpha*K*C));
    end
end

