% Exp 3) Repeat Exp 2) for cancer dataset with two different optimisers of your choice e.g.
% ‘trainlm’ and ‘trainrp’. Comment and discuss the result and decide which is more appropriate
% training algorithm for the problem. In your discussion, include in your description a detailed
% account of how the training algorithms (optimisations) work


% ensemble ensemble = 11
% changing number of nodes/epochs
NODES = 2:5:32;
EPOCHS = 1:3:64;
N = 10;
ensemble_size = 23;
[x,t] = cancer_dataset;

for epochs = 1: length(EPOCHS)
    for nodes = 1:length(NODES)
    % net_ensemble.m
    [train_ensemble_avg_errors(epochs, nodes), test_ensemble_avg_errors(epochs, nodes)] = net_ensemble(x, t, epochs, nodes, ensemble_size);
    end
end

for i = 1:length(NODES)
    train_min_node(i) = min(train_ensemble_avg_errors(:,i));
    test_min_node(i) = min(test_ensemble_avg_errors(:,i));
    train_index(i) = find(train_ensemble_avg_errors(:,i)==train_min_node(i),1);
    test_index(i) = find(test_ensemble_avg_errors(:,i)==test_min_node(i),1);
end

% Plots
% train error
figure()
for i = 1:size(train_ensemble_avg_errors, 2)
    plot(EPOCHS ,train_ensemble_avg_errors(:,i));
    hold on
end
title(sprintf('Ensemble Avg Classification Error Training ensemble= %d', ensemble_size))
xlabel('Epochs')
ylabel('Classification Error in %')
legend([num2str(NODES(1)), ' nodes',', min error ',num2str(train_min_node(1)),' at epoch ', num2str(train_index(1))], ...
       [num2str(NODES(2)), ' nodes',', min error ',num2str(train_min_node(2)),' at epoch ', num2str(train_index(2))], ...
       [num2str(NODES(3)), ' nodes',', min error ',num2str(train_min_node(3)),' at epoch ', num2str(train_index(3))], ...
       [num2str(NODES(4)), ' nodes',', min error ',num2str(train_min_node(4)),' at epoch ', num2str(train_index(4))], ...
       [num2str(NODES(5)), ' nodes',', min error ',num2str(train_min_node(5)),' at epoch ', num2str(train_index(5))], ...
       [num2str(NODES(6)), ' nodes',', min error ',num2str(train_min_node(6)),' at epoch ', num2str(train_index(6))], ...
       [num2str(NODES(7)), ' nodes',', min error ',num2str(train_min_node(7)),' at epoch ', num2str(train_index(7))])
hold off

% test error
figure()
for i = 1:size(test_ensemble_avg_errors, 2)
    plot(EPOCHS ,test_ensemble_avg_errors(:,i));
    hold on
end
title(sprintf('Ensemble Avg Classification Error Testing ensemble= %d', ensemble_size))
xlabel('Epochs')
ylabel('Classification Error in %')
legend([num2str(NODES(1)), ' nodes',', min error ',num2str(test_min_node(1)),' at epoch ', num2str(test_index(1))], ...
       [num2str(NODES(2)), ' nodes',', min error ',num2str(test_min_node(2)),' at epoch ', num2str(test_index(2))], ...
       [num2str(NODES(3)), ' nodes',', min error ',num2str(test_min_node(3)),' at epoch ', num2str(test_index(3))], ...
       [num2str(NODES(4)), ' nodes',', min error ',num2str(test_min_node(4)),' at epoch ', num2str(test_index(4))], ...
       [num2str(NODES(5)), ' nodes',', min error ',num2str(test_min_node(5)),' at epoch ', num2str(test_index(5))], ...
       [num2str(NODES(6)), ' nodes',', min error ',num2str(test_min_node(6)),' at epoch ', num2str(test_index(6))], ...
       [num2str(NODES(7)), ' nodes',', min error ',num2str(test_min_node(7)),' at epoch ', num2str(test_index(7))])
hold off
   
   
   