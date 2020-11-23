% author: Rui Wang
% date: 2019
% copyright@ JNU_B411 (Jiangnan University)
% department: school of artificial intelligence and computer science (AI&CS)
% Note, the values of the key parameters used in this paper need to be fine-tuned according to the datasets you created and used 

clear;
close all;
clc;
%step1 load data.mat
load demo-ETH
load S_ab   % random values, we select a suitable combination of a and b. For random situation, the final acc. will be fluctuated

%step2: make labels for training data and test data
Train_lables=zeros(1,40);
Test_lables=zeros(1,40);
%labels for training data
l=5;
k=l;
a=linspace(1,8,8); % 8 means the number of categories
i1=1;
while(k<=40)
    while(i1<=8)
        for i=1:8
            i_train=l*(i-1)+1;
            Train_lables(i_train:k)=a(i1);
            k=k+5;
            i1=i1+1;
        end
    end
end

% labels for test data
l1=5;
k1=l1; 
a1=linspace(1,8,8);
i2=1;
while(k1<=40)
    while(i2<=8)
        for i=1:8
            i_test=l1*(i-1)+1;
            Test_lables(i_test:k1)=a(i2);
            k1=k1+5;
            i2=i2+1;
        end
    end
end

param.d=400; % original dimension
d=param.d;
basis=eye(d); % 

data_train=cell(1,40);
data_test=cell(1,40);
accuracy_matrix=zeros(1,10); %

% to check which one is appropriate for the proposed algorithm,57-59 used to store the random values for selection
r_a = cell(1, 10);
r_b = cell(1,10);
r_v = cell(1,10);
a_each = ab.A;
b_each = ab.B;
V0 = ab.V;

for iteration = 1:1
        
    data_train = ETH_train;    
    data_test =  ETH_test;     
    
    train_Gras = cell(1,40); % training Grassmannian data 
    log_cov_train_Spd = cell(1,40); % use to store log-Euclidean of the training samples
    
    test_Gras = cell(1,40); % test Grassmannian data
    log_cov_test_Spd = cell(1,40); % use to store log-Euclidean of the test samples
    
    %% step5: computing COV, log-map, and Gaussian embedded model for training samples
    tic
    [ls_train, q1] = compute_sub(data_train); % obtaining the Grassmann manifold-valued data
    cov_train = compute_cov(data_train); % obtaining the SPD manifold-valued data
    SGM_train = Compute_SGM(data_train); % caculate single Gaussian model for each train data set
    % this part is applied to get the Gaussian embedded model for each SGM of training
    G_train = Compute_SGM_Embedding(SGM_train); 
    
    % this part is mainly used to obtain the Gras data and cov_log data for training
    for i=1:40
        
        temp_tr_Gras = ls_train{i};
        temp_tr_Spd = cov_train{i};
        log_cov_train_Spd{i} = logm(temp_tr_Spd); 
        train_Gras{i} = temp_tr_Gras; 
        
    end   
    toc
    disp('get train data')
    
    %% step6: computing COV, log-map, and Gaussian embedded model for test samples
    tic
    [ls_test, q2] = compute_sub(data_test); % 
    cov_test = compute_cov(data_test);
    SGM_test = Compute_SGM(data_test); % caculate single Gaussian model for each test data set, it is a row cell
    
    % this part is applied to get the embedded Gaussian model for each SGM of testing
    G_test = Compute_SGM_Embedding(SGM_test); 
    
    for i=1:40
        
        temp_te_Gras = ls_test{i};
        temp_te_Spd = cov_test{i};
        test_Gras{i} = temp_te_Gras; 
        log_cov_test_Spd{i} = logm(temp_te_Spd); 
        
    end
    toc
    disp('get test data')
    
    %% step7: building the training and test kernel matrices
    kmatrix_train = zeros(size(train_Gras,2),size(train_Gras,2)); % Grassmannian training kernel 
    kmatrix_test = zeros(size(train_Gras,2),size(test_Gras,2)); % Grassmannian test kernel
    
    kmatrix_train_Spd = zeros(size(log_cov_train_Spd,2),size(log_cov_train_Spd,2)); % SPD training kernel
    kmatrix_test_Spd = zeros(size(log_cov_train_Spd,2),size(log_cov_test_Spd,2)); % SPD test kernel 
    
    %calculate linear kernel of single Gaussian model
    tic
    kmatrix_train_Gau = Compute_Riemann_Kernel_Gau(G_train,[]); % use the Eq.14 in the corresponding PR paper to generate train kernel
    toc
    disp('Gau kernel train')
    tic
    kmatrix_test_Gau = Compute_Riemann_Kernel_Gau(G_train, G_test); % use the Eq.14 in the corresponding PR paper to generate test kernel
    toc
    disp('Gau kernel test')
    kmatrix_train_Gau = kmatrix_train_Gau / 100000;
    kmatrix_test_Gau = kmatrix_test_Gau / 100000;
    
    tic
    for i = 1:size(train_Gras,2)
        for j = 1:size(train_Gras,2)
            cov_i_Train = train_Gras{i}; 
            cov_j_Train = train_Gras{j}; 
            temp_i = cov_i_Train * cov_i_Train';
            temp_j = cov_j_Train*cov_j_Train';
            temp_i = temp_i(:);
            temp_j = temp_j(:);
            kmatrix_train(i,j) = temp_i' * temp_j; % trace((cov_i_Train*cov_i_Train')*(cov_j_Train*cov_j_Train')); %141*141
            kmatrix_train(j,i) = kmatrix_train(i,j);
        end
    end
    toc
    disp('Grass kernel train')
    kmatrix_train = kmatrix_train / 100000;
    
    tic
    for i = 1:size(log_cov_train_Spd,2)
        for j = 1:size(log_cov_train_Spd,2)
            cov_i_Train = log_cov_train_Spd{i}; 
            cov_j_Train = log_cov_train_Spd{j}; 
            cov_i_Train_reshape = reshape(cov_i_Train,size(cov_i_Train,1)*size(cov_i_Train,2),1); 
            cov_j_Train_reshape = reshape(cov_j_Train,size(cov_j_Train,1)*size(cov_j_Train,2),1);
            kmatrix_train_Spd(i,j) = cov_i_Train_reshape'*cov_j_Train_reshape; % 141*141
            kmatrix_train_Spd(j,i) = kmatrix_train_Spd(i,j);
        end
    end
    toc
    disp('SPD kernel train')
    kmatrix_train_Spd = kmatrix_train_Spd / 100000;
    
    tic
    for i=1:size(train_Gras,2)
        for j=1:size(test_Gras,2)
            cov_i_Train=train_Gras{i}; 
            cov_j_Test=test_Gras{j}; 
            temp_i = cov_i_Train * cov_i_Train';
            temp_j = cov_j_Test*cov_j_Test';
            temp_i = temp_i(:);
            temp_j = temp_j(:);
            kmatrix_test(i,j)=temp_i'*temp_j; % trace((cov_i_Train*cov_i_Train')*(cov_j_Test*cov_j_Test')); % 240*141
        end
    end
    toc
    disp('Grass kernel test')
    kmatrix_test = kmatrix_test / 100000;
    
    tic
    for i=1:size(log_cov_train_Spd,2)
        for j=1:size(log_cov_test_Spd,2)
            cov_i_Train=log_cov_train_Spd{i}; 
            cov_j_Test=log_cov_test_Spd{j}; 
            cov_i_Train_reshape=reshape(cov_i_Train,size(cov_i_Train,1)*size(cov_i_Train,2),1); 
            cov_j_Test_reshape=reshape(cov_j_Test,size(cov_j_Test,1)*size(cov_j_Test,2),1);
            kmatrix_test_Spd(i,j)=cov_i_Train_reshape'*cov_j_Test_reshape;
        end
    end
    toc
    disp('SPD kernel test')
    kmatrix_test_Spd = kmatrix_test_Spd / 100000;
    
    %%%%%%% the above is fixed, next is to design the kernel learning method, we need to adjust the following matlab code
    %% kernel normalization
    % lamda1 = 0.8; % for Gras kernel feature
    % lamda2 = 0.2;% for Spd kernel feature
    alpha = 5e-2; % the balance parameter of objective function (10-12)  5e-2
    
    %% Compute the core matrix U
    tic
    [ U, a_all, b_all, rand_a_all, rand_b_all ] = multi_kernel_metric_learning(kmatrix_train, kmatrix_train_Spd, kmatrix_train_Gau, Train_lables, alpha, a_each, b_each, V0);
    toc
    disp('Training')
    dist = zeros(size(Train_lables,2),size(Test_lables,2)); % dist matrix
    a1 = a_all(:,1);
    a2 = a_all(:,2);
    a3 = a_all(:,3);
    b1 = b_all(:,1);
    b2 = b_all(:,2);
    b3 = b_all(:,3);
    r_a{iteration} = rand_a_all;
    r_b{iteration} = rand_b_all;

%% Classification
tic
for i_dist=1:size(Train_lables,2)
    Y_train_gras = kmatrix_train(:,i_dist); 
    Y_train_spd = kmatrix_train_Spd(:,i_dist); 
    Y_train_sgm = kmatrix_train_Gau(:,i_dist); 
    value_gating_func_sum_left = exp(a1'*Y_train_gras+b1) + exp(a2'*Y_train_spd+b2) + exp(a3'*Y_train_sgm+b3);
    lamda1_l = exp(a1'*Y_train_gras+b1) / value_gating_func_sum_left;
    lamda2_l = exp(a2'*Y_train_spd+b2) / value_gating_func_sum_left;
    lamda3_l = exp(a3'*Y_train_sgm+b3) / value_gating_func_sum_left;
     for j_dist=1:size(Test_lables,2)
         Y_test_gras = kmatrix_test(:,j_dist); 
         Y_test_spd = kmatrix_test_Spd(:,j_dist); 
         Y_test_sgm = kmatrix_test_Gau(:,j_dist);
         value_gating_func_sum_right = exp(a1'*Y_test_gras+b1) + exp(a2'*Y_test_spd+b2) + exp(a3'*Y_test_sgm+b3);
         lamda1_r = exp(a1'*Y_test_gras+b1) / value_gating_func_sum_right;
         lamda2_r = exp(a2'*Y_test_spd+b2) / value_gating_func_sum_right;
         lamda3_r = exp(a3'*Y_test_sgm+b3) / value_gating_func_sum_right;
         Y_dist1 = lamda1_l * (Y_train_gras-Y_test_gras)' * U * U' * (Y_train_gras-Y_test_gras) * lamda1_r;
         Y_dist2 = lamda2_l * (Y_train_spd-Y_test_spd)' * U * U' * (Y_train_spd-Y_test_spd) * lamda2_r;
         Y_dist3 = lamda3_l * (Y_train_sgm-Y_test_sgm)' * U * U' * (Y_train_sgm-Y_test_sgm) * lamda3_r;
         dist(i_dist,j_dist) = Y_dist3 + Y_dist1 + Y_dist2;
     end
end
 toc
 disp('Classification')
 
 test_num = size(Test_lables,2); % number of test samples
 [dist_sort,index] = sort(dist,1,'ascend'); 
 % right_num=length(find((Test_labels'-Train_labels'(index(1,:)))==0));
 right_num = length(find((Test_lables'-Train_lables(index(1,:))')==0));
 accuracy = right_num/test_num; 
 accuracy_matrix(iteration) = accuracy*100;
 fprintf(1,'the number of right recognized samples of the %d-th iteration is£º%d\n',iteration, right_num );
 fprintf(1,'the classification score of the %d-th iteration is: %d %d\n', iteration ,accuracy*100);

end
mean_accuracy = sum(accuracy_matrix) / 1.0;
fprintf(1,'average classification score is: %d\n',mean_accuracy);
fprintf(1,'std is: %d\n',std(accuracy_matrix));
