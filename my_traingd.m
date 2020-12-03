function [train_ensemble_avg_errors, test_ensemble_avg_errors] = my_traingd(x, t, classifiers, nodes, epochs)
    N = 10;
    hiddenSizes = nodes;
    trainFcn = 'trainlm';
    net = feedforwardnet(hiddenSizes);

    net.trainFcn = 'traingd';
    net.trainParam.epochs = epochs;
    net.input.processFcns = {'removeconstantrows','mapminmax'};
    % Divide data similarly for ensemble
    net.divideFcn = 'divideint'; 
    % Divide up every sample
    net.divideMode = 'sample';
    net.divideParam.trainRatio = 50/100;
    net.divideParam.valRatio = 0/100;
    net.divideParam.testRatio = 50/100;
    net.performFcn = 'crossentropy';

    % Ensemble:
    % multiple classifiers with random starting weights.
    for c = 1:length(classifiers)
         for n = 1:N
                for i = 1:classifiers(c)
                % concat, suffle dataset and split again
                dataset = [x; t];
                dataset_shuffled = dataset(:, randperm(size(dataset, 2)));
                x_shuffled = dataset_shuffled(1:9,:);
                t_shuffled = dataset_shuffled(10:11,:);

                % Train the Network
                [net,tr] = train(net,x_shuffled,t_shuffled);

                % split data to train data
                train_x = x(:,tr.trainInd );
                train_t = t(:,tr.trainInd );
                train_t_class = vec2ind(train_t);
                % predict
                train_y = net(train_x);
                train_y_class(i,:) = vec2ind(train_y);

                % split data to test data
                test_x = x(:,tr.testInd );
                test_t = t(:,tr.testInd );
                test_t_class = vec2ind(test_t);
                % predict
                test_y = net(test_x);
                test_y_class(i,:) = vec2ind(test_y);

                % Test the Network
                performance = perform(net,train_t,train_y);
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

        train_ensemble_avg_errors(c) = mean(train_ensemble_errors);
        test_ensemble_avg_errors(c) = mean(test_ensemble_errors);
    end