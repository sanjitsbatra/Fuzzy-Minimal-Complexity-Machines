function [ trainAcc, testAcc, nsv, time ] = Fuzzy_MCM(xTrain, yTrain, xTest, yTest, C, beta)

    tic

    N=size(xTrain,1);
    D=size(xTrain,2);

    %%% Define Kernel
    % Linear
%     Kernel = @(x,y) ( (x*y'));
    % RBF
    Kernel = @(x,y) exp(-beta * norm(x-y)^2);    
    
    
    % Generate Fuzzy memberships
    S = ones(N,1);

    delta = 0.05;
    
    xp(:,:) = xTrain(find(yTrain == 1),:);
    xm(:,:) = xTrain(find(yTrain == -1),:);
    xp_mean = mean(xp);
    xm_mean = mean(xm);

    rp = 0;
    rm = 0;
    for i = 1:size(xp,1)
      r_temp = sqrt(sum((xp_mean - xp(i,:)) .^ 2));
      if(r_temp > rp)
          rp = r_temp;
      end
    end
    for i = 1:size(xm,1)
      r_temp = sqrt(sum((xm_mean - xm(i,:)) .^ 2));
      if(r_temp > rm)
          rm = r_temp;
      end
    end    

    
    for i = 1:N
        if(yTrain(i) == 1)
            r = sqrt(sum((xp_mean - xTrain(i,:)) .^ 2));
            S(i) = (1 - (r/(rp+delta)));
        else
            r = sqrt(sum((xm_mean - xTrain(i,:)) .^ 2));
            S(i) = (1 - (r/(rm+delta)));
        end
    end
                                                           
            
    
    %solve linear program
    X = [randn(N,1);randn(1,1);randn(N,1);randn(1,1)];       %[lambda, b, q, h]
    f = [zeros(N,1);zeros(1,1);C*S;1];    

   LM = zeros(N,N);
    for i=1:N
        for j=1:N
            LM(i,j) = yTrain(i) * Kernel(xTrain(i,:),xTrain(j,:));
        end
    end
    
    
%   [lambda,               b,          q,              h]
A = [ LM        ,     yTrain,   eye(N,N),   -1*ones(N,1);
     -LM        ,    -yTrain,  -eye(N,N),     zeros(N,1);];

B = [zeros(N,1);-1*ones(N,1);];


Aeq = [];
Beq = [];

%    [        lambda,      b,             q,     h]
lb = [-inf*ones(N,1);   -inf;    zeros(N,1);     0;];
ub = [ inf*ones(N,1);    inf; inf*ones(N,1);   inf;];

options=optimset('display','none', 'Largescale', 'off', 'Simplex', 'on');

[X, fval, exitflag]  = linprog(f,A,B,Aeq,Beq, lb,ub, [], options);
lambda = X(1:N,:);
b = X(N + 1,:);
q = X(N+1 + 1:N+1 + N,:);
h = X(2*N+1 + 1,:);

    yPredTrain = yTrain*0;
    
    for i = 1:N
        sumj = b;
        for j = 1:N
            sumj = sumj + lambda(j) * Kernel(xTrain(j, :), xTrain(i, :));
        end
        yPredTrain(i) = sumj;
    end

    trainAcc = sum(yPredTrain.*yTrain>0)/size(yTrain,1) * 100;

    Ntest = size(xTest, 1);
    yPredTest = yTest*0;

    for i = 1:Ntest
        sumj = b;
        for j = 1:N
            sumj = sumj + lambda(j) * Kernel(xTrain(j, :), xTest(i, :));
        end
        yPredTest(i) = sumj;
    end

    testAcc = sum(yPredTest.*yTest>0)/size(yTest,1) * 100;    

  
    nsv = 0;
    for k = 1:N
        if(lambda(k)~=0)
            nsv = nsv +1;
        end
    end
    
    time = toc;

    fprintf(2, 'Fuzzy MCM\nTraining set accuracy: %f \t Test set accuracy: %f \t nsv: %d \t Time = %f \n', trainAcc, testAcc, nsv, time);

end
