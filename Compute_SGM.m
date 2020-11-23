function SGMs = Compute_SGM(X)
%calculate single Gaussian model of eath set data

nSam = length(X);
SGMs = cell(1, nSam);

for i = 1 : nSam % 
    
    DataSet_i = X{i}/255;   
    % EM calculation
    [~ , SGModel_i , ~] = emgm(DataSet_i,1);
    SGMs{i}.mu = SGModel_i.mu; % mean value
    SGMs{i}.R = SGModel_i.Sigma; % cov
    SGMs{i}.w = SGModel_i.weight; % prior probability 
    
end
