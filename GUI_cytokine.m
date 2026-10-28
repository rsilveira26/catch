%function [out] = algo_pso_cytokine
%UNTITLED Summary of this function goes here
% Clear workspace
clear; clc; close all; warning off;

    % Ensure dataset is in the working directory

    %variablename=readmatrix('Dataset.csv');
    variablename=readmatrix('Dataset.csv');

    

    % Replace with actual dataset filename
    % data = readtable('Dataset.csv');
    data = readtable('Dataset.csv');


% Access data and labels (assuming first column is label, rest are data)

data_features = variablename(:,3:47);  
labels = data.Class;  
labels=strcmp(labels, 'Control');

tic
msvm_results=evl(data_features,labels,'MSVM')
msvm_time=toc
disp('---------------------------------------')

tic
svm_results=evl(data_features,labels,'SVM')
svm_time=toc
disp('---------------------------------------')

    save('cytokine.mat','variablename');


    leng_data = size(data);
    isEqual = strcmp(data.Class, 'Control');
    variablenames = horzcat(variablename, double(isEqual));
    X = variablenames(:,3:47); % All columns except last (features)
    y=variablenames(:,49); % Last column is target variable

    % Convert categorical variables if needed
    data.Gender = categorical(data.Gender);
    data.Class = categorical(data.Class);
    X = normalize(X);

    tic

    % Normalize features
    % Set PSO parameters
    numParticles = 10;     % Number of particles
    numIterations = 50;    % Number of iterations
    numFeatures = size(X, 2);
    w = 0.7;               % Inertia weight
    c1 = 1.5;             % Cognitive component
    c2 = 1.5;             % Social component


    % Initialize particles (binary for feature selection)
    particles = randi([0 1], numParticles, numFeatures);
    velocities = rand(numParticles, numFeatures) * 0.1 - 0.05; % Small random velocities
    pBest = particles;  % Personal best positions
    gBest = particles(1, :); % Global best position
    pBestScores = ones(numParticles, 1) * Inf; % Personal best scores
    gBestScore = Inf;  % Best score globally

    % Fitness function: Evaluate classification accuracy
    function acc = fitnessFunction(selectedFeatures, X, y)
    selectedIdx = find(selectedFeatures == 1);
    if isempty(selectedIdx)  % Avoid empty feature subset
        acc = 0;
        return;
    end
    X_selected = X(:, selectedIdx);
    cv = cvpartition(size(X_selected, 1), 'HoldOut', 0.2);
    X_train = X_selected(training(cv), :);
    y_train = y(training(cv), :);
    X_test = X_selected(test(cv), :);
    y_test = y(test(cv), :);

    mdl = fitglm(X_train, y_train, 'Distribution', 'normal');
    y_pred = round(predict(mdl, X_test));
    acc = sum(y_pred == y_test) / length(y_test);
    end

    % PSO Optimization Loop
    for iter = 1:numIterations
        for i = 1:numParticles
            % Evaluate fitness (accuracy)
            fitness = fitnessFunction(particles(i, :), X, y);

            % Update personal best
            if fitness > pBestScores(i)
                pBestScores(i) = fitness;
                pBest(i, :) = particles(i, :);
            end

            % Update global best
            if fitness > gBestScore
                gBestScore = fitness;
                gBest = particles(i, :);
            end
        end

        % Update particle velocity and position
        for i = 1:numParticles
            r1 = rand(1, numFeatures);
            r2 = rand(1, numFeatures);
            velocities(i, :) = w * velocities(i, :) + ...
                c1 * r1 .* (pBest(i, :) - particles(i, :)) + ...
                c2 * r2 .* (gBest - particles(i, :));

            % Apply sigmoid to ensure binary (feature selection)
            probabilities = 1 ./ (1 + exp(-velocities(i, :)));
            particles(i, :) = probabilities > rand(1, numFeatures);
        end
    end

    % Final selected features
    selectedFeatures = find(gBest == 1);
    %disp('Selected Features Index:');
    %disp(selectedFeatures);

    % Train final logistic regression model with selected features
    if isnan(selectedFeatures)
            X_final = X(:, :);

    else
    X_final = X(:, selectedFeatures);
    end
    cv = cvpartition(size(X_final, 1), 'HoldOut', 0.2);
    X_train = X_final(training(cv), :);
    y_train = y(training(cv), :);


    X_test = X_final(test(cv), :);
    y_test = y(test(cv), :);

    mdl_final = fitglm(X_train, y_train, 'Distribution', 'normal');
    save mdl_final
    y_pred = round(predict(mdl_final, X_test));

    % Evaluate model
    finalAccuracy = sum(y_pred == y_test) / length(y_test);
    %disp(['Final Model Accuracy with PSO: ', num2str(finalAccuracy * 100), '%']);

    % Confusion Matrix
    confMat = confusionmat(y_test, y_pred);
   % disp('Confusion Matrix:');
    TP=confMat(1,1)
    TN=confMat(2,2)
    FP=confMat(1,2)
    FN=confMat(2,1)

    accuracy=(TP+TN)/(TP+TN+FP+FN)*100;
    precision= TP/ (TP + FP)*100;
    if isnan(precision)
        precision=0*100;
    end
    recall= TP/ (TP + FN)*100;
    if isnan(recall)
        recall=0*100;
    end

    %end

    F1score=(2*precision*recall)/(precision+recall);
    dt_time=toc;
    disp(['Average: Selected Features  Accuracy Precision  Recall  F1Score  Time  ' ]);
    out=[length(selectedFeatures ), accuracy,precision,recall,F1score,dt_time]

%end
  

%disp(confMat);
% 
% disp('---------------------------------------')
