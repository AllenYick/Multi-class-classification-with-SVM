% main
clear,clc
load iris_class1_2_3_4D.mat;
n_fold = 5;   %number of folds
method = 2;  %replace R_1 by 0, 1, 2 or 3 according to the method you use
[DX, Dt, R] = PR_SVM_subdata( X, t, n_fold, method );
C_values = [1,10,100];   %values of C 
G = 2;
types =[ 'linear','poly','RBF']; % kernel_types
accuracy = cell(3,3);

for i = 1 : 3
    C = C_values(i);
    for kernel_type = 0 : 2
        train_params = ['-s 0 -t ' num2str(kernel_type) ' -g ' num2str(G) ' -r 0 -c ' num2str(C)];
        accuracy{i,kernel_type+1} = binary_decision_tree(DX,Dt,n_fold,train_params);
    end
end



           
            
           
        

