function Y1_T = Compute_SGM_Embedding(SY1)
%calculate embedding model for each single Gaussian model 

number_sets1=length(SY1); % the length in this dataset is 40
Y1_T = cell(1, number_sets1); % used to store the embedded model and its corresponding weights
for tmpC1 = 1:number_sets1
    Y1 = SY1{tmpC1}; % get each SGM
    Y1_mu = Y1.mu; % get mean value
    Y1_cov = Y1.R; % get standar derivation
    num_gau = size(Y1_mu,2); % its num is 1, because of single Gaussian
    Y1_T{tmpC1}.T = cell(1, num_gau);
    Y1_T{tmpC1}.W = Y1.w; % get weights
    for i = 1 : size(Y1_mu,2) % the result is 1
        y1_m = Y1_mu(:,i);         
        y1_c = Y1_cov(:,:,i);        
        lamda = 0.001 * trace(y1_c); 
        y1_c = y1_c + lamda * eye(size(y1_c,1)); % regularization        
        
        %embedding of Gaussian component 
        % y1_t = det(y1_c)^(-1/(size(y1_c,1)+1))*[y1_c+y1_m*y1_m' y1_m;y1_m' 1];
        y1_t = [y1_c+y1_m*y1_m' y1_m;y1_m' 1];    
        Y1_T{tmpC1}.T{i} = logm(y1_t); %logarithm
        
    end   
end