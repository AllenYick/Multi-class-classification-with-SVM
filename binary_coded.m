function [accuracy] = binary_coded(DX,Dt,n_fold,params)
%METHOD Binary Decision Tree
%   ACCURACY = BINARY_CODED(X,T,N_FOLD,PARAMS)
%   DX: data, DT: lable or class
%   N_FOLD: number of fold, PARAMS: training parameters
%   accuracy is training accuracy and testing accuracy of every process

accuracy = zeros(5,2);
for i = 1 : n_fold
    %-- training
    train_data = DX{i, 3};   %% class 1,2(+1) and class 3(-1)
    train_class = Dt{i, 3};
    model(1) = svmtrain(train_class, train_data,params );
    train_data = DX{i, 4};   %% class 1,3(+1) and class 2(-1)
    train_class = Dt{i, 4};
    model(2) = svmtrain(train_class, train_data,params );

    %-- testing
    for j = 1 : 2
        test_data  = DX{i, j};
        test_class = Dt{i, j};
        dims = size(test_class,1);
        %-- for every sample in test_data
        [predict_lable_1, ~, ~] = svmpredict(rand(dims,1) , test_data, model(1),'-q' );
        [predict_lable_2, ~, ~] = svmpredict(rand(dims,1) , test_data, model(2),'-q' );
        predict_lable_1(predict_lable_1==1) = 0;
        predict_lable_1(predict_lable_1==-1) = 2;
        predict_lable_2(predict_lable_2==1) = 0;
        predict_lable_2(predict_lable_2==-1) = 1;
        predict_class = predict_lable_1 + predict_lable_2;
        accuracy(i,j) = sum( predict_class==test_class ) / dims;
    end
end

