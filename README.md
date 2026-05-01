function [Alpha_score,Alpha_pos,Convergence_curve] = DMSGWO(SearchAgents_no,Max_Iteration,lb,ub,dim,fobj)

    Alpha_pos=zeros(1,dim);
    Alpha_score=inf; 
    Beta_pos=zeros(1,dim);
    Beta_score=inf; 
    Delta_pos=zeros(1,dim);
    Delta_score=inf; 

    Positions = initialization(SearchAgents_no, dim, ub, lb); 

    n1 = round(SearchAgents_no * 0.5); 
    n2 = SearchAgents_no - n1;     
    
    no_improvement_count = 0;   
    previous_best_score = inf; 

    Convergence_curve=zeros(1,Max_Iteration); 


    t=0;
    while t<Max_Iteration
        gwo_fit = zeros(SearchAgents_no, 1);

        for i=1:size(Positions,1)
            Flag4ub=Positions(i,:)>ub;
            Flag4lb=Positions(i,:)<lb;
            Positions(i,:)=(Positions(i,:).*(~(Flag4ub+Flag4lb)))+ub.*Flag4ub+lb.*Flag4lb;           
            
            gwo_fit(i)=fobj(Positions(i,:));  

            if gwo_fit(i)<Alpha_score 
                Alpha_score=gwo_fit(i);   % Update alpha
                Alpha_pos=Positions(i,:);
            end
            if gwo_fit(i)>Alpha_score && gwo_fit(i)<Beta_score 
                Beta_score=gwo_fit(i);    % Update beta
                Beta_pos=Positions(i,:);
            end  
            if gwo_fit(i)>Alpha_score && gwo_fit(i)>Beta_score && gwo_fit(i)<Delta_score 
                Delta_score=gwo_fit(i);   % Update delta
                Delta_pos=Positions(i,:);
            end
        end

        a = 2/log10(2) * log10(2-(t/Max_Iteration)^3);

        f_avg = mean(gwo_fit);

        for i = 1:n1
            for j = 1:dim
                p = rand();
                if p > 0.5
                    newpop = Positions(i,:)+rand()*(Positions(i,j)-Alpha_pos(j)).*Levy(dim);
                else
                    rand_indices = randperm(SearchAgents_no, 2);
                    X_random1 = Positions(rand_indices(1), j);
                    X_random2 = Positions(rand_indices(2), j);
                    newpop = Positions(i,:)+ rand()*(X_random1-X_random2);
                end
                Flag4ub = newpop>ub;
                Flag4lb = newpop<lb;
                newpop = (newpop.*(~(Flag4ub+Flag4lb)))+ub.*Flag4ub+lb.*Flag4lb;

                newfit = fobj(newpop);
                if newfit < gwo_fit(i)
                    gwo_fit(i) = newfit;
                    Positions(i,:) = newpop;
                end
                
            end
        end

        for i = n1+1:SearchAgents_no
            for j = 1:dim
                r1 = rand(); 
                r2 = rand();
                A1 = 2 * a * r1 - a;
                C1 = 2 * r2;
                D_alpha = abs(C1 * Alpha_pos(j) - Positions(i,j));
                  X1 = Alpha_pos(j) - A1 * D_alpha;

                r1 = rand();
                r2 = rand();
                A2 = 2 * a * r1 - a;
                C2 = 2 * r2;
                D_beta = abs(C2 * Beta_pos(j) - Positions(i,j));
                  X2 = Beta_pos(j) - A2 * D_beta;
    
                r1 = rand();
                r2 = rand();
                A3 = 2 * a * r1 - a;
                C3 = 2 * r2;
                D_delta = abs(C3 * Delta_pos(j) - Positions(i,j));
                  X3 = Delta_pos(j) - A3 * D_delta;


                    X = (X1 + X2 + X3) / 3;
                    theta = 3;
                    epsilon = 1e-8;
                    phi = (gwo_fit(i) / (f_avg + epsilon))^theta;
                    w = 0.5 + 0.5 * cos(pi * t / Max_Iteration); 
                    Positions(i,j) = (1 - w) * X +  w * X * phi;
            end
        end

            if Alpha_score < previous_best_score     
                no_improvement_count = 0; 
                if n1 > 1 && n2 < SearchAgents_no - 1 
                    n1 = n1 - 1;
                    n2 = n2 + 1;
                end
            else  
                no_improvement_count = no_improvement_count + 1; 
                if no_improvement_count >= 3 
                    if n1 < SearchAgents_no - 1 && n2 > 1 
                        n1 = n1 + 1;
                        n2 = n2 - 1;
                    end  
                else  
                    if n1 > 1 && n2 < SearchAgents_no - 1
                        n1 = n1 - 1;
                        n2 = n2 + 1;
                    end
                end
            end

        previous_best_score = Alpha_score;

        t=t+1;
        Convergence_curve(t)=Alpha_score; 

    end
end

function step = Levy(d)

    lambda = 3/2; 

    sigma =(  gamma(1+lambda)*sin(pi*lambda/2) / (gamma((1+lambda)/2)*lambda*2^((lambda-1)/2))  )^(1/lambda);   %gamma(X)返回在 X 的元素处计算的 gamma 函数值。
    u = randn(1,d)*sigma;
    v = randn(1,d);

    step = u./abs(v).^(1/lambda);

end


function Positions=initialization(SearchAgents_no,dim,ub,lb)

    Boundary_no= size(ub,2); % numnber of boundaries

    % If the boundaries of all variables are equal and user enter a signle
    % number for both ub and lb
    if Boundary_no==1
        Positions=rand(SearchAgents_no,dim).*(ub-lb)+lb;
    end

    % If each variable has a different lb and ub
    if Boundary_no>1
        for i=1:dim
            ub_i=ub(i);
            lb_i=lb(i);
            Positions(:,i)=rand(SearchAgents_no,1).*(ub_i-lb_i)+lb_i;
        end
    end
end

