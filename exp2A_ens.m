% Exp 2A)

% Choose an optimal number of nodes and epochs based on Exp1.
% Ensemble of individual classifiers with random starting weights
% and Majority votes. Repeat 30 times with different 50/50 splits.
% Graph the average result.
% Repeat for a different number of individual classifiers (3-25), and
% comment the difference between ensemble and individual accuracy with
% number of classifiers variation.
% Consider changing the number of epochs and nodes to see the difference in
% accuracy. Comment the difference in optimal nodes/epochs between an
% ensemble and a base classifier.

% Need to use different NN and select the majority of the predicted class
% per observation as the final prediction.
nodes = 8;
epochs = 16;
N = 10;
% ensemble sizes (odd numbers).
classifiers = 3:2:25;

% Majority  vote of individual classifiers
[x,t] = cancer_dataset;
trainFcn = 'trainscg';  % Scaled conjugate gradient backpropagation.
hiddenLayerSize = nodes;
net = patternnet(hiddenLayerSize, trainFcn);
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

train_Errors_std = mean(std(train_ensemble_avg_errors));
test_Errors_std = mean(std(test_ensemble_avg_errors));

% Plots
% train error
figure()
plot(classifiers ,train_ensemble_avg_errors)
hold on
% test error
plot(classifiers ,test_ensemble_avg_errors)
legend('train','test')
title('Training and Test Error vs Ensemble size')
xlabel('Number of classifiers per Ensemble')
ylabel('Error (%)')
hold off


