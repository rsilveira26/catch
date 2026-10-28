clc
clear 
close all
variablename=readmatrix('Dataset.csv');
data = readtable('Dataset.csv');
save('cytokine.mat','variablename');
% features = data{:, 1:3};  % This will get all rows for columns 1, 2, 3
% 
% % Extract labels (assuming the last column is the label)
% labels = data{:, 4};  % This will get all rows for column 4
leng_data = size(data);
isEqual = strcmp(data.Class, 'Control');
variablenames = horzcat(variablename, double(isEqual));
data_use = variablenames(:,3:47);
labels=variablenames(:,49);
tic
msvm_results=evl(data_use,labels,'MSVM')
msvm_time=toc
disp('---------------------------------------')
tic
svm_results=evl(data,labels,'SVM')
svm_time=toc
disp('---------------------------------------')
% 
% tic
% knn_results=evl(data,labels,'KNN')
% knn_time=toc
% disp('---------------------------------------')

tic
dt_results=evl(data,labels,'DT')
dt_time=toc
disp('---------------------------------------')

tic
nb_results=evl(data,labels,'NB')
nb_time=toc
disp('---------------------------------------')

% tic
% mlp_results=evl(data,labels,'NN')
% mlp_time=toc
% disp('---------------------------------------')

tic
bag_results=evl(data,labels,'BAG')
bag_time=toc
disp('---------------------------------------')

tic
boost_results=evl(data,labels,'Boost')
boost_time=toc
disp('---------------------------------------')

tic
rand_results=evl(data,labels,'RAND')
rand_time=toc