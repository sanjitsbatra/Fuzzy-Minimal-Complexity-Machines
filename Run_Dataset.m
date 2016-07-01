function [time] = Run_Dataset(filename, label_type, label_last, nfolds)
% runs a given dataset
% generates folds for a given dataset 
% prepares the data into training and testing folds.


data = load(strcat('Data/',filename));
if (label_last == 1)
x=data(:,1:end-1);
y=data(:,end);
else
x=data(:,2:end);
y=data(:,1);
end

if (label_type == 1)
y(find(y <= 0)) = -1;
y(find(y > 0)) = 1;
end

if (label_type == 2)
y(find(y <= 1)) = -1;
y(find(y > 1)) = 1;
end

D=size(x,2);

%fix missing values
for d=1:D,
    col=x(:,d);
    m=mean(col(find(col==col)));
    col(find(col~=col))=m;
    x(:,d)=col;
end

% random permutation
rand_seed = 1;
rand('state',rand_seed);
r=randperm(size(y,1));
y=y(r,:);
x=x(r,:);

nsize = size(x, 1);
foldstart = 1 + round(nsize * (0:nfolds)/nfolds);

%--------------------------------------------------------------------------


ffuzzy = fopen(strcat('Results/Fuzzy_',filename(1:end-5),'.txt'), 'w');
fprintf(ffuzzy, '%s \n\n\n', filename);
fprintf(ffuzzy, 'Format: \nSeed,Fold,C,beta,Trainacc,Testacc,nsv,time\n\n\n');

fsvm = fopen(strcat('Results/SVM_',filename(1:end-5),'.txt'), 'w');
fprintf(fsvm, '%s \n\n\n', filename);
fprintf(fsvm, 'Format: \nSeed,Fold,C,beta,Trainacc,Testacc,nsv,time\n\n\n');

fsummary = fopen(strcat('Results/Summary_',filename(1:end-5),'.txt'), 'a');
fprintf(fsummary, '%s \n\n\n', filename);
fprintf(fsummary, 'Format: \nSeed,Fold,C,beta,Trainacc,Testacc,nsv,time\n\n\n');
fprintf(fsummary, 'Order: \nFuzzy\nSVM\n\n\n');
%--------------------------------------------------------------------------

% Paramater Tuning
Clist = [1e-5,1e-3,1e-1,1e0,1e1,1e3,1e5];
betalist = [0.0001,0.0005,0.001,0.005,0.01,0.05,0.1,0.5,1,5,10];

counter = 0;

for Ci = 1:1%length(Clist)
    for betaj = 1:2%length(betalist)
        C = Clist(Ci);
        beta = betalist(betaj);

%--------------------------------------------------------------------------
    counter = counter + 1;

for fold = 1:nfolds
    s = foldstart(fold);
    e = foldstart(fold + 1) - 1;
    xTest = x((s:e), :);
    yTest = y((s:e), :);
    %fprintf(2, 'Test(fold = %d): s e %d %d \n', fold, s, e);
    e1 = foldstart(fold) - 1;
    s2 = foldstart(fold + 1);
    e2 = nsize;
    %fprintf(2, 'Train(fold = %d): s1 e1 %d %d s2 e2 %d %d \n', fold, 1, e1, s2, e2);
    xTrain = [x(1:e1, :); x(s2:e2, :)];
    yTrain = [y(1:e1, :); y(s2:e2, :)];
    
    %centre data
    m = mean(xTrain);
    stdev = std(xTrain);
    for d=1:D%centre data
        if(stdev(d)~=0)
            xTrain(:,d) = (xTrain(:,d) - m(d))/stdev(d);
            xTest(:, d) = (xTest(:, d) - m(d))/stdev(d);
        else     
            xTrain(:,d) = (xTrain(:,d) - m(d));
            xTest(:, d) = (xTest(:, d) - m(d));
        end        
    end
    
%--------------------------------------------------------------------------

    % Generate stats_files for each fold
    % Write the following for each dataset:
    % For each fold, save the fold, the C,beta,Kernel, nSV or nW, and Acc

    if(counter == 1)
    % Save the xTrain,yTrain,xTest,yTest in a mat file, only once
    save(strcat('Results/',filename(1:end-5),'_',int2str(fold),'_Data.mat'),'xTrain','yTrain','xTest','yTest');
    end

    fprintf(2,'\nDataset: %s \t Fold: %d\n',filename(1:end-5),fold);
    
    % Fuzzy
    [ trainacc, testacc, nsv, time ] = Fuzzy_MCM(xTrain, yTrain, xTest, yTest, C, beta);
    
    fuzzy_trainAcc(fold) = trainacc;
    fuzzy_testAcc(fold) = testacc;
    fuzzy_nSV(fold) = nsv;
    fuzzy_Time(fold) = time;

    fprintf(ffuzzy, '%d, %d, %f, %f, %f, %f, %d, %f\n', rand_seed, fold, C, beta, trainacc,testacc,nsv,time);
    
    % End of fold stats are represented with fold = -1
    eofold = -1;
    if(fold == nfolds)
      fprintf(ffuzzy, '%d, %d, %f, %f, %f +- %f, %f +- %f, %f +- %d, %f +- %f\n', rand_seed, eofold, C, beta, mean(fuzzy_trainAcc),std(fuzzy_trainAcc),mean(fuzzy_testAcc),std(fuzzy_testAcc),mean(fuzzy_nSV),std(fuzzy_nSV),mean(fuzzy_Time),std(fuzzy_Time));
      fprintf(fsummary, '\n%d, %d, %f, %f, %f +- %f, %f +- %f, %f +- %d, %f +- %f\n', rand_seed, eofold, C, beta, mean(fuzzy_trainAcc),std(fuzzy_trainAcc),mean(fuzzy_testAcc),std(fuzzy_testAcc),mean(fuzzy_nSV),std(fuzzy_nSV),mean(fuzzy_Time),std(fuzzy_Time));
    end

    
%     % SVM
%     [ trainacc, testacc, nsv, time ] = SVM(xTrain, yTrain, xTest, yTest, C, beta);
%     
%     svm_trainAcc(fold) = trainacc;
%     svm_testAcc(fold) = testacc;
%     svm_nSV(fold) = nsv;
%     svm_Time(fold) = time;
% 
%     fprintf(fsvm, '%d, %d, %f, %f, %f, %f, %d, %f\n', rand_seed, fold, C, beta, trainacc,testacc,nsv,time);
%     
%     % End of fold stats are represented with fold = -1
%     eofold = -1;
%     if(fold == nfolds)
%       fprintf(fsvm, '%d, %d, %f, %f, %f +- %f, %f +- %f, %d +- %f, %f +- %f\n', rand_seed, eofold, C, beta, mean(svm_trainAcc),std(svm_trainAcc),mean(svm_testAcc),std(svm_testAcc),mean(svm_nSV),std(svm_nSV),mean(svm_Time),std(svm_Time));
%       fprintf(fsummary, '\n%d, %d, %f, %f, %f +- %f, %f +- %f, %f +- %f, %f +- %f\n', rand_seed, eofold, C, beta, mean(svm_trainAcc),std(svm_trainAcc),mean(svm_testAcc),std(svm_testAcc),mean(svm_nSV),std(svm_nSV),mean(svm_Time),std(svm_Time));
%     end
% 
%--------------------------------------------------------------------------

% We need to display summary information for each choice of parameter
% values, for the cross validation, for each dataset. Format:


%--------------------------------------------------------------------------

end
    end
   
end

   
  
  
       
   
     
end


