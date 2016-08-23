function [F, W, b, i, objFinal] = LPE(X, P, S, alpha, beta, gamma)

 % min F,W,b>=0 ||X'*W+1n*b'-F||^2_F + alpha*||S(F-P)||^2_F + beta||W|| +
 % r||W'W-I||

    % input - known variables:
        % X = data, d*n
        % P = perception over instances, n*p (only seeds have values)
        % S = diagonal seed indicator, n*n
    % unknown
        % W = d*p (mapping matrix: mapp original feature space to perception space) 
        % b = p*1 (intercept vector)
        % F = perception feature space, n*p
        
	stop = 0.00005;
	maxIter = 10000;
		
   % initialed outside
     [d, n] = size(X); 
     [n, p] = size(P);
    
	
   W = rand(d,p);
   b = rand(p,1);
   F =  rand(n,p);
   oneN = ones(n,1);


    
    obj(1) = objective(W, b, F);
    
    i=0;
    
    while 1
        i=i+1;
        
        %if(mod(i,2000)==0) 
        %    i 
        %    obj(i)
        %end
        
        [gradF] = funF(F);
        Fnew = gradF;
            
        [gradW] = funW(W);        
        Wnew = gradW;
        
        
        [gradb] = funb(b);
        bnew = gradb;
        
        obj(i+1)= objective(Wnew, bnew, Fnew);
     %  disp( obj(i+1));

        if ((obj(i)-obj(i+1)>=0 && obj(i)-obj(i+1)<stop) )
           %plot(obj(100:end));
           %saveas(gca,['g' num2str(i)],'bmp');
           break; 
        end
        if (i+1>maxIter)
        %   plot(obj(100:end));
        %   saveas(gca,['g' num2str(i)],'bmp');
        %    i=0;
            break;
        end
        

        b = bnew;
        W = Wnew;
        F = Fnew;

    end
    objFinal = obj(length(obj));
    
    
    function [obj] = objective(W, b, F)
        obj = norm(X'*W+oneN*b'-F,'fro') + alpha*norm(S*F-S*P,'fro') + beta*norm(W,'fro') + gamma*norm(W'*W-eye(p),'fro');
    end
    
    function [grad] = funF(F) 
        grad = F .* sqrt((X'*W+oneN*b'+alpha*S*P)./(F+alpha*S*F));
    end

    function [grad] = funb(b)
        grad = b .* sqrt((F'*oneN)./(W'*X*oneN+b*oneN'*oneN));
    end

    function [grad] = funW(W)
        grad = W .* sqrt((X*F+gamma*W)./(X*X'*W+X*oneN*b'+beta*W+gamma*W*W'*W));
    end

end

