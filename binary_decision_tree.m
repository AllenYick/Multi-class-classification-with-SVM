function [accuracy] = binary_decision_tree(DX,Dt,n_fold,params)
%METHOD Binary Decision Tree
%   ACCURACY = BINARY_DECISION_TREE(X,T,N_FOLD,PARAMS)
%   DX: data, DT: lable or class
%   N_FOLD: number of fold, PARAMS: training parameters
%   accuracy is training accuracy and testing accuracy of every process

accuracy = zeros(5,2);
for i = 1 : n_fold
    %-- training
    train_data = DX{i, 3};   %% class 1,2(+1) and class 3(-1)
    train_class = Dt{i, 3};
    model(1) = svmtrain(train_class, train_data,params );
    train_data = DX{i, 4};   %% class 1(+1) and class 2(-1)
    train_class = Dt{i, 4};
    model(2) = svmtrain(train_class, train_data,params );

    %-- testing
    for j = 1 : 2
        test_data  = DX{i, j};
        test_class = Dt{i, j};
        dims = size(test_class,1);
        predict_class = zeros(dims,1);   % 1,2or 3
        %-- for every sample in test_data
        for k = 1 : dims
             %-- first classifier
            [predict_lable, ~, ~] = svmpredict(rand , test_data(k,:), model(1),'-q' );
            if predict_lable == 1
                 %-- second classifier
                [predict_lable, ~, ~] = svmpredict(rand , test_data(k,:), model(2),'-q' );
                if predict_lable == 1
                    predict_class(k) = 1;
                else
                    predict_class(k) = 2;
                end
            else  
                predict_class(k) = 3;
            end
        end
        test_class = test_class + 1;
        accuracy(i,j) = sum( predict_class==test_class ) / dims;
    end
end

