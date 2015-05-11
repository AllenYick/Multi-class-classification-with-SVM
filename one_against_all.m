function [accuracy] = one_against_all(DX,Dt,n_fold,params)
%one against one
accuracy = zeros(5,2);
for i = 1 : n_fold
    %-- training
    train_data = DX{i, 3};   %% class 1(+1) and class 2&3(-1)
    train_class = Dt{i, 3};
    model(1) = svmtrain(train_class, train_data,params );
    train_data = DX{i, 4};   %% class 2(+1) and class 1&3(-1)
    train_class = Dt{i, 4};
    model(2) = svmtrain(train_class, train_data,params );
    train_data = DX{i, 5};   %% class 3(+1) and class 2&3(-1)
    train_class = Dt{i, 5};
    model(3) = svmtrain(train_class, train_data,params );
    %-- testing
    for j = 1 : 2
        test_data  = DX{i, j};
        test_class = Dt{i, j};
        dims = size(test_class,1);
        test_lable = rand(dims,1);   %random
        vote = zeros(dims,3);
        decisions = zeros(dims,3);
        %-- model 1     
        [~, ~, decision_values] = svmpredict(test_lable , test_data, model(1),'-q' );
        decisions(:,1) = decision_values;
        %-- model 2
        [~, ~, decision_values] = svmpredict(test_lable, test_data, model(2),'-q' );
        decisions(:,2) = decision_values;
        %-- model 3
        [~, ~, decision_values] = svmpredict(test_lable, test_data, model(3),'-q' );
        decisions(:,3) = decision_values;
        [~, ind] = max( transpose(decisions) );
        test_class = test_class' + 1;
        accuracy(i,j) = sum(ind==test_class)/dims;
    end
end

