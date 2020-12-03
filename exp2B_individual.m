% exp2B_individual

[x,t] = cancer_dataset;

% train and test error
% varie number of nodes and epochs
n_nodes = [4, 8, 16, 32, 64];
n_epochs = [8, 16, 32, 64, 128];
% net_individual.m
for i = 1:length(n_nodes)
    for nodes = n_nodes(i)
        for j  = 1:length(n_epochs)
            for epochs = n_epochs(j)
           % individual classifier function
           [train_ind_avg_errors(i, j), test_ind_avg_errors(i, j)] = net_individual(x, t, nodes, epochs);
            end
        end
    end
end

figure()
plot(n_epochs, train_ind_avg_errors)
legend('4 nodes','8 nodes','16 nodes','32 nodes','64 nodes')
xlabel('epochs')
ylabel('Error in %')
title('Individual classifier Train Error vs Epochs')
hold off

figure()
plot(n_epochs, test_ind_avg_errors)
legend('4 nodes','8 nodes','16 nodes','32 nodes', '64 nodes')
xlabel('epochs')
ylabel('Error in %')
title('Individual classifier Test Error vs Epochs')
hold off
