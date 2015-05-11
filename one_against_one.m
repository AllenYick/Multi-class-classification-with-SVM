function [accuracy] = one_against_one(DX,Dt,n_fold,params)
%one against one
accuracy = zeros(5,2);
for i = 1 : n_fold
    %-- training
    train_data = DX{i, 3};   %% class 1(+1) and class 2(-1)
    train_class = Dt{i, 3};
    model(1) = svmtrain(train_class, train_data,params );
    train_data = DX{i, 4};   %% class 1(+1) and class 3(-1)
    train_class = Dt{i, 4};
    model(2) = svmtrain(train_class, train_data,params );
    train_data = DX{i, 5};   %% class 2(+1) and class 3(-1)
    train_class = Dt{i, 5};  
    model(3) = svmtrain(train_class, train_data,params );
    %-- testing
    for j = 1 : 2
        test_data  = DX{i, j};
        test_class = Dt{i, j};
        dims = size(test_class,1);
        test_lable = rand(dims,1);
        vote = zeros(dims,3);
        %-- model 1
        [predict_lable_test, ~,prob_estimates] = svmpredict(test_lable , test_data, model(1),'-q' );
         %-- vote
        for m = 1 : dims
            if predict_lable_test(m) == 1
                vote(m,1) = vote(m,1)+1;
            else
                vote(m,2) = vote(m,2)+1;
            end
        end
       %-- model 2
       [predict_lable_test, ~,prob_estimates] = svmpredict(test_lable, test_data, model(2),'-q' );
       %-- vote
        for m = 1 : dims
            if predict_lable_test(m) == 1
                vote(m,1) = vote(m,1)+1;
            else
                vote(m,3) = vote(m,3)+1;
            end
        end
      
       %-- model 3
       [predict_lable_test, ~,prob_estimates] = svmpredict(test_lable, test_data, model(3),'-q' );
      %-- vote
        for m = 1 : dims
            if predict_lable_test(m) == 1
                vote(m,2) = vote(m,2)+1;
            else
                vote(m,3) = vote(m,3)+1;
            end
        end 
        [~, ind] = max( transpose(vote) );
        test_class = test_class' + 1;
        accuracy(i,j) = sum(ind==test_class)/dims;
    end
end

