% Use Majority of votes to select output class of each test and traning
% object.
function [train_ind_avg_errors, test_ind_avg_errors]  = net_individual(x, t, nodes, epochs)

 N = 10;
    for n = 1:N
    trainFcn = 'trainscg';  % Scaled conjugate gradient backpropagation.
    hiddenLayerSize = nodes;
    net = patternnet(hiddenLayerSize, trainFcn);
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

    % Train the Network (the train function does the train/val/test split)
    [net,tr] = train(net,x_shuffled,t_shuffled);

    % train data with split made by train()
    train_x = x_shuffled(:,tr.trainInd );
    train_t = t_shuffled(:,tr.trainInd );
    train_t_class = vec2ind(train_t);
    % predict
    train_y = net(train_x);
    train_y_class = vec2ind(train_y);

    % test data
    test_x = x_shuffled(:,tr.testInd );
    test_t = t_shuffled(:,tr.testInd );
    test_t_class = vec2ind(test_t);
    % predict
    test_y = net(test_x);
    test_y_class = vec2ind(test_y);

    % train_error
    train_ind_errors(n) = sum(train_t_class ~= train_y_class)/numel(train_t_class);

    % test_error
    test_ind_errors(n) = sum(test_t_class ~= test_y_class)/numel(test_t_class);
    end
 train_ind_avg_errors = mean(train_ind_errors);
 test_ind_avg_errors = mean(test_ind_errors);
 net = init(net);
end