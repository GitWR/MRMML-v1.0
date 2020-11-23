function cov_t = compute_cov(data)

  num_t = length(data); 
  cov_t = cell(1,num_t);
  
for i = 1:num_t
    
  sample_t = data{i} / 255; 
  sample_t_mean = mean(sample_t, 2);
  sample_t_c = sample_t - repmat(sample_t_mean,1,size(sample_t,2));
  cov_temp = (sample_t_c * sample_t_c')/(size(sample_t,2)-1);
  cov_t{i} = cov_temp + trace(cov_temp) * (1e-3) * eye(size(cov_temp,1));
  
end
end

