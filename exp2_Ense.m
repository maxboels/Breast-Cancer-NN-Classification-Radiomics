% changing number of nodes/epochs
nodes = 2:6:32;
epochs = 1:3:64;
N = 10;
ensemble = 23;

[x,t] = cancer_dataset;
trainFcn = 'trainscg';  % Scaled conjugate gradient backpropagation.
for e = 1: length(epochs) % 1:22
    for n = 1:length(nodes) % 1:6
    net = patternnet(nodes(n), trainFcn);
    net.trainParam.epochs = epochs(e);

    net.input.processFcns = {'removeconstantrows','mapminmax'};
    net.divideFcn = 'divideint';  % Divide data similarly for ensemble
    net.divideMode = 'sample';  % Divide up every sample
    net.divideParam.trainRatio = 50/100;
    net.divideParam.valRatio = 0/100;
    net.divideParam.testRatio = 50/100;
    net.performFcn = 'crossentropy';  % Cross-Entropy

    train_y_class = zeros(1,350);
    test_y_class = zeros(1,349);
    
    % net_E11.m
    [train_ensemble_avg_errors(e, n), test_ensemble_avg_errors(e, n)] = net_Ense(net, x, t, ensemble);
    end
end


figure()
% Plots
% train error
for i = 1:size(train_ensemble_avg_errors, 2)
    plot(epochs ,train_ensemble_avg_errors(:,i));
    hold on
end
title(sprintf('Ensemble Avg Classification Error Training ensemble= %d',ensemble))
xlabel('Epochs')
ylabel('Classification Error in %')
legend([num2str(nodes(1)), ' nodes'], [num2str(nodes(2)), ' nodes'], ...
       [num2str(nodes(3)), ' nodes'], [num2str(nodes(4)), ' nodes'], ...
       [num2str(nodes(5)), ' nodes'], [num2str(nodes(6)), ' nodes'])
hold off

figure()
% test error
for i = 1:size(test_ensemble_avg_errors, 2)
    plot(epochs ,test_ensemble_avg_errors(:,i));
    hold on
end
title(sprintf('Ensemble Avg Classification Error Testing ensemble= %d',ensemble))
xlabel('Epochs')
ylabel('Classification Error in %')
legend([num2str(nodes(1)), ' nodes'], [num2str(nodes(2)), ' nodes'], ...
       [num2str(nodes(3)), ' nodes'], [num2str(nodes(4)), ' nodes'], ...
       [num2str(nodes(5)), ' nodes'], [num2str(nodes(6)), ' nodes'])
hold off
   
   
   