function [sub_t , q_value] = compute_sub(data)

  num_t=length(data); 
  sub_t=cell(1,num_t);
  
for i=1:num_t
    
  sample_t_c = data{i}/255; 
  cov_t = sample_t_c * sample_t_c';
  [U,~,~] = svd(cov_t); 
  q_value = 40; % needs to be adjusted according to your CV problems and datasets
  sub_t{i} = U(:,1:q_value); 
   
end
end

