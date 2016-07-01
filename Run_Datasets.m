% runs a set of datasets and compiles results into a file

clear;clc;close all

nfolds = 5;
time = zeros(nfolds, 1);



% cd C:\Users\Win7_02\Documents\Sanjit\LS\Dual\
cd ~/Desktop/Play/ML_JD/MCM/Fuzzy' Classifier'/


% [time] = Run_Dataset('pimaindiansdiabetes.data', 1, 1, nfolds);
% [time] = Run_Dataset('heart_statlog.data', 2, 1, nfolds);
% [time] = Run_Dataset('haberman.data', 2, 1, nfolds);
% [time] = Run_Dataset('hepatitis.data', 2, 0, nfolds);
% [time] = Run_Dataset('ionosphere.data', 2, 1, nfolds);
% [time] = Run_Dataset('transfusion.data', 1, 1, nfolds);
% [time] = Run_Dataset('tictactoe.data', nfolds);
% [time] = Run_Dataset('echocardiogram.csv', 1, 1, nfolds);
% [time] = Run_Dataset('promoters.csv', 2, 1, nfolds);
% [time] = Run_Dataset('voting.txt', 2, 0, nfolds);
% [time] = Run_Dataset('horsecolic.csv', 2, 0, nfolds);
% [time] = Run_Dataset('fertility_Diagnosis.csv', 1, 1, nfolds);
% [time] = Run_Dataset('australian.csv', 1, 1, nfolds);
% [time] = Run_Dataset('crx.csv', 1, 0, nfolds);
% [time] = Run_Dataset('mammographic_masses.data', 1, 1, nfolds);
% [time] = Run_Dataset('german.csv', 2, 1, nfolds);
% [time] = Run_Dataset('bands.csv', 1, 1, nfolds);
% [time] = Run_Dataset('breastw.data', 2, 1, nfolds);
[time] = Run_Dataset('plrx.csv', 2, 1, nfolds);
% [time] = Run_Dataset('sonar_data.data', 2, 1, nfolds);
% [time] = Run_Dataset('load house_votes_84.csv', 1, 0, nfolds);


fclose all;