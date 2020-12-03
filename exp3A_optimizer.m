% Exp 3A)

% use 2 different optimizers from [traingd: Gradient descent backpropagation
%                                  , trainlm: Levenberg-Marquardt backpropagation
%                                  , trainrp: Resilient backpropagation]

% before we used trainscg: Scaled conjugate gradient backpropagation
nodes = 22;
epochs = 21;
classifiers = 3:2:25;

% Majority  vote of individual classifiers
[x,t] = cancer_dataset;

% optimizer function traingd.m
[traingd_ensemble_avg_errors, testgd_ensemble_avg_errors] = my_traingd(x, t, classifiers, nodes, epochs);

% optimizer function trainlm.m
[trainlm_ensemble_avg_errors, testlm_ensemble_avg_errors] = my_trainlm(x, t, classifiers, nodes, epochs);

% Plots
% traingd error
figure()
plot(classifiers ,traingd_ensemble_avg_errors)
hold on
plot(classifiers ,testgd_ensemble_avg_errors)
hold on
plot(classifiers ,trainlm_ensemble_avg_errors)
hold on
plot(classifiers ,testlm_ensemble_avg_errors)
legend('train GD','test GD', 'train LM', 'test LM')
title('Training and Test Error vs Ensemble size')
xlabel('Number of classifiers per Ensemble')
ylabel('Error (%)')

