% Usage: [DX, Dt, R] = PR_SVM_subdata(X, t, n_fold, method)
%
% Input:
%   - X: instance matrix
%   - t: label vector
%   - n_fold: number of folds for cross validation
%   - method: types of multi-class classification
%       0 -- one-against-one
%       1 -- one-against-all
%       2 -- binary decision tree
%       3 -- binary coded 
%
% Output:
%   - DX: instance matrix in sub-data sets
%   - Dt: label vector in sub-data sets
%   - R: number of classes in the dataset X
% where DX and Dt are cell structures:
%       - DX{i, 1}, Dt{i, 1}: all training data of n_fold. The "i" in {i,
%       1} indicates the i-th fold training dataset.  The "1" in {i,1} indicates all
%       training data except the one left out.  Fro example, when n_fold =
%       5, the program will divide the whole dataset (150 data) into 5
%       sub-datasets denoted as D1, D2, D3, D4 and D5. Each of them consists of 30 data.
%       The output will give DX{1,1} = [D2, D3, D4, D5], DX{2,1} = [D1, D3, D4, D5],
%       DX{3,1} = [D1, D2, D4, D5], DX{4,1} = [D1, D2, D3, D5], DX{5,1} =
%       [D1, D2, D3, D4]. Each DX{i,1} consists of 120 data. Dt{i,1} is the
%       corresponding label vector for DX{i,1}.
%
%       - DX{i, 2}, Dt{i, 2}: test data (the one left out).  The "i" in {i,2} 
%       indicates the i-th fold test dataset.  The "2" in {i,2}
%       indicates all test dataset except the one left out.  For example,
%       when n_fold = 5 , DX{1,2} = [D1], DX{2,2} = [D2], DX{3,2} = [D3], 
%       DX{4,2} = [D4] and DX{5,2} = [D5]. Each DX{i,2} consists of 30 data. 
%       Dt{i,2} is the corresponding label vector for DX{i,2}.
%
%       - DX{i, 3}, Dt{i, 3}: training data for class 1 (re-label from '0' to '+1') vs class 2 (re-label from '1' to '-1') when method = 0; 
%                                           for class 1 (+1) vs classes 2&3 (-1) when method = 1; 
%                                           for class 1&2 (+1) vs class 3 (-1) when method = 2; 
%                                           for class 1&2 (+1) vs class 3 (-1) when method = 3; 
%
%       - DX{i, 4}, Dt{i, 4}: training data for class 1 (+1) vs class 3 (-1) when method = 0; 
%                                           for class 2 (+1) vs classes 1&3 (-1) when method = 1; 
%                                           for class 1 (+1) vs class 2 (-1) when method = 2; 
%                                           for class 1&3 (+1) vs class 2 (-1) when method = 3; 
%
%       - DX{i, 5}, Dt{i, 5}: training data for class 2 (+1) vs class 3 (-1) when method = 0; 
%                                           for class 3 (+1) vs classes 1&2 (-1) when method = 1;
%                                           not used when method = 2, 3;
% where i = 1,2,...,n_fold, is the i^th time cross-validation process 
%
% Dt has the same structure as Dx.  The only difference is that Dt stores the class labels.

function [DX, Dt, R] = PR_SVM_subdata(X, t, n_fold, method)

[row1, col1] = size(X);
if row1 < col1
    X = X';
end
[row2, col2] = size(t);
if row2 < col2
    t = t';
end

[C,~,ic] = unique(t); % C = t(it); t = C(ic); find all classes and corresponding indices
R = length(C); % number of classes

for i = 1:R
    n_sample_c(i) = sum(ic == i); % number of samples in each class
    tt((1:n_sample_c(i)) + sum(n_sample_c(1:(i - 1))), 1) = t(ic == i, 1); % group the samples of the same class together, just in case they are separate in the data; if already together, tt = t, XX = X
    XX((1:n_sample_c(i)) + sum(n_sample_c(1:(i - 1))), :) = X(ic == i, :);
    p_index{i} = randperm(n_sample_c(i)); % randomly permuted indices for cross validation
end

s_fold = n_sample_c/n_fold; % size of each class in each fold

for i = 1:n_fold

    training_data_index = [];
    testing_data_index = [];
    for j = 1:R
        training_data_index = [training_data_index, p_index{j}(1:(s_fold(j)*(i - 1))) + sum(n_sample_c(1:(j - 1))), p_index{j}((s_fold(j)*i + 1):end) + sum(n_sample_c(1:(j - 1)))]; % rest of the sub-data indicis
        testing_data_index = [testing_data_index, p_index{j}((s_fold(j)*(i - 1) + 1):(s_fold(j)*i)) + sum(n_sample_c(1:(j - 1)))]; % D_i sub-data indicis, it is guaranteed that it includes 's_fold(j)' samples of class j
    end
    
    Dt{i, 1} = tt(training_data_index, 1); % rest of the sub-data %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    DX{i, 1} = XX(training_data_index, :);
    
    Dt{i, 2} = tt(testing_data_index, 1); % D_i sub-data %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    DX{i, 2} = XX(testing_data_index, :);
    
    [~,~,ic1] = unique(Dt{i, 1});
    
    switch(method)
        
        case(0) % one-against-one
            for k = 1:R
                for l = (k + 1):R
                    Dt{i, k + l} = [ones(sum(ic1 == k), 1); - ones(sum(ic1 == l), 1)]; % relabel to +-1; smaller sub-data for training %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    DX{i, k + l} = [DX{i, 1}(ic1 == k, :);  DX{i, 1}(ic1 == l, :)];
                end
            end
            
            
        case(1) % one-against-all
            for k = 1:R
                Dt{i, k + 2} = [ones(sum(ic1 == k), 1); - ones(sum(ic1 ~= k), 1)]; % relabel to +-1; smaller sub-data for training %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                DX{i, k + 2} = [DX{i, 1}(ic1 == k, :);  DX{i, 1}(ic1 ~= k, :)];
            end
            
            
        case(2) % binary decision tree; only for R = 3
            % first level: 12|3
            Dt{i, 3} = [ones(sum(ic1 <= 2), 1); - ones(sum(ic1 == 3), 1)]; % relabel to +-1; smaller sub-data for training %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            DX{i, 3} = [DX{i, 1}(ic1 <= 2, :);  DX{i, 1}(ic1 == 3, :)];

            % second level: 1|2
            Dt{i, 4} = [ones(sum(ic1 == 1), 1); - ones(sum(ic1 == 2), 1)]; % relabel to +-1; smaller sub-data for training %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            DX{i, 4} = [DX{i, 1}(ic1 == 1, :);  DX{i, 1}(ic1 == 2, :)];
            

        case(3) % binary coded approach; only for R = 3
            % SVM1: 12|3
            Dt{i, 3} = [ones(sum(ic1 <= 2), 1); - ones(sum(ic1 == 3), 1)]; % relabel to +-1; smaller sub-data for training %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            DX{i, 3} = [DX{i, 1}(ic1 <= 2, :);  DX{i, 1}(ic1 == 3, :)];

            % SVM2: 13|2
            Dt{i, 4} = [ones(sum(ic1 ~= 2), 1); - ones(sum(ic1 == 2), 1)]; % relabel to +-1; smaller sub-data for training %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            DX{i, 4} = [DX{i, 1}(ic1 ~= 2, :);  DX{i, 1}(ic1 == 2, :)];
            
    end

end






