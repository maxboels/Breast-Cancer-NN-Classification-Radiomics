% functio for classifier = E and changing nodes/epcohs

function [train_ensemble_avg_errors, test_ensemble_avg_errors] = net_ensemble(x, t, epochs, nodes, ensemble_size)
N = 5; % avg
Ensemble = ensemble_size; % ensemble size
    for n = 1:N
            for e = 1:Ensemble
            trainFcn = 'trainscg';  % Scaled conjugate gradient backpropagation.
            net = patternnet(nodes, trainFcn);
            net.trainParam.epochs = epochs;
            net.input.processFcns = {'removeconstantrows','mapminmax'};
            net.divideFcn = 'divideint';  % Divide data similarly for ensemble
            net.divideMode = 'sample';  % Divide up every sample
            net.divideParam.trainRatio = 50/100;
            net.divideParam.valRatio = 0/100;
            net.divideParam.testRatio = 50/100;
            net.performFcn = 'crossentropy';  % Cross-Entropy

            % concat, suffle dataset and split again
            dataset = [x; t];
            dataset_shuffled = dataset(:, randperm(size(dataset, 2)));
            x_shuffled = dataset_shuffled(1:9,:);
            t_shuffled = dataset_shuffled(10:11,:);

            % Train the Network
            [net,tr] = train(net,x_shuffled,t_shuffled);

            %train data
            train_x = x(:,tr.trainInd );
            train_t = t(:,tr.trainInd );
            train_t_class = vec2ind(train_t);
            % predict
            train_y = net(train_x);
            train_y_class(e,:) = vec2ind(train_y);

            % test data
            test_x = x(:,tr.testInd );
            test_t = t(:,tr.testInd );
            test_t_class = vec2ind(test_t);
            % predict
            test_y = net(test_x);
            test_y_class(e,:) = vec2ind(test_y);

            net = init(net);
            end
    
    % majority.m train set to 1d vector
    train_class_vote = majority(train_y_class, length(train_y_class));
    % train_errors from c ensembles
    train_ensemble_errors(n,:) = sum(train_t_class ~= train_class_vote)/numel(train_t_class);

    % majority.m test set
    test_class_vote = majority(test_y_class, length(test_y_class)); 
    % test_errors from c ensembles
    test_ensemble_errors(n,:) = sum(test_t_class ~= test_class_vote)/numel(test_t_class);
    end
train_ensemble_avg_errors = mean(train_ensemble_errors);
test_ensemble_avg_errors = mean(test_ensemble_errors);
end
    
    
    