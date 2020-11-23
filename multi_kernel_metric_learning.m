function [ U, a_all, b_all, a_rand_all, b_rand_all ] = multi_kernel_metric_learning( k_Gras, k_Spd, k_Sgm, Train_lables, alpha, a_each, b_each, V0)

  num_class = length(unique(Train_lables)); % 
  D = size(k_Gras,1); % depend on the used dataset
  d = 8; % needs to be adjusted according to your CV datasets and problems
  itera = 25; 
  itera_innner = 9;
  %V0 = rand(D, d);
  V_compute = cell(1,itera_innner); % used to justify when stopping to optimize U 
  
%   a1 = rand(D,1); % gating function of Gras
%   a2 = rand(D,1); % corresponding to SPD
%   a3 = rand(D,1); % corresponding to SGM
%   
%   b1 = rand(1,1); % correspond to Gras
%   b2 = rand(1,1); % corresponding to SPD
%   b3 = rand(1,1); % corresponding to SGM
  
  a1 = a_each.a1;
  b1 = b_each.b1(1,:);
  
  a2 = a_each.a2;
  b2 = b_each.b2(1,:);
  
  a3 = a_each.a3;
  b3 = b_each.b3(1,:);
  
  a_rand_all = zeros(D, 3);
  b_rand_all = zeros(D, 3);
  
  %% easy to check which one is suitable for our method
  a_rand_all(:,1) = a1;
  a_rand_all(:,2) = a2;
  a_rand_all(:,3) = a3;
  
  b_rand_all(:,1) = b1;
  b_rand_all(:,2) = b2;
  b_rand_all(:,3) = b3;
  
  %% .........
  a_1 = zeros(D,itera);
  a_2 = zeros(D,itera);
  a_3 = zeros(D,itera);
  b_1 = zeros(1,itera);
  b_2 = zeros(1,itera);
  b_3 = zeros(1,itera);
  
  a_1(:,1) = a1;
  a_2(:,1) = a2;
  a_3(:,1) = a3;
  
  b_1(:,1) = b1;
  b_2(:,1) = b2;
  b_3(:,1) = b3;
  
  a_all = zeros(D,3); % used to store each a that corresponding to each model
  b_all = zeros(1,3); % used to store each b that corresponding to each model
  
for i = 1 : itera
    
  fprintf('\n i= %d \n',i);
  Sw=zeros(D,D); % 141 * 141
  Sb=zeros(D,D); % 141 * 141
  % for intra class
  a1_dev_sum = zeros(D,1);
  a2_dev_sum = zeros(D,1); 
  a3_dev_sum = zeros(D,1);
  
  b1_dev_sum = 0;
  b2_dev_sum = 0;
  b3_dev_sum = 0;
  % for inter class
  a1_dev_sum_inter = zeros(D,1);
  a2_dev_sum_inter = zeros(D,1);
  a3_dev_sum_inter = zeros(D,1);
  
  b1_dev_sum_inter = 0;
  b2_dev_sum_inter = 0;
  b3_dev_sum_inter = 0;
  
  Nw = 0;
  Nb = 0;
  
  for j = 1 : num_class
      num_eachclass = find(Train_lables==j); 
      for k = 1 : length(num_eachclass)
          K_gras_data1 = k_Gras(:,num_eachclass(k));
          K_spd_data1 = k_Spd(:,num_eachclass(k));
          K_sgm_data1 = k_Sgm(:,num_eachclass(k));
         %% gating function computing-->denominator-->left side 
          temp_gating_func_sum_left = exp(a1'*K_gras_data1+b1) + exp(a2'*K_spd_data1+b2) + exp(a3'*K_sgm_data1+b3);
         %% caculate each model's weight of left side
          yita1_l = exp(a1'*K_gras_data1+b1) / temp_gating_func_sum_left;
          yita2_l = exp(a2'*K_spd_data1+b2)  / temp_gating_func_sum_left;
          yita3_l = exp(a3'*K_sgm_data1+b3)  /  temp_gating_func_sum_left;
          for m=k+1 : length(num_eachclass)
              K_gras_data2 = k_Gras(:,num_eachclass(m));
              K_spd_data2 = k_Spd(:,num_eachclass(m));
              K_sgm_data2 = k_Sgm(:,num_eachclass(m));
            %% gating function computing--> molecule-->right side
              temp_gating_func_sum_right = exp(a1'*K_gras_data2+b1) + exp(a2'*K_spd_data2+b2) + exp(a3'*K_sgm_data2+b3);
            %% caculate each model's weight of right side
              yita1_r = exp(a1'*K_gras_data2+b1) / temp_gating_func_sum_right;
              yita2_r = exp(a2'*K_spd_data2+b2)  / temp_gating_func_sum_right;
              yita3_r = exp(a3'*K_sgm_data2+b3)  / temp_gating_func_sum_right;
            %% caculate the inner part of the dev(Rw)/dev(a)
              % step1, for a1
              part1_dev_a_each1 = yita1_l * (K_gras_data1-K_gras_data2) * (K_gras_data1-K_gras_data2)' * yita1_r ;
              part1_dev_a_each2 = yita2_l * (K_spd_data1-K_spd_data2) * (K_spd_data1-K_spd_data2)' * yita2_r ;
              part1_dev_a_each3 = yita3_l * (K_sgm_data1-K_sgm_data2) * (K_sgm_data1-K_sgm_data2)' * yita3_r ;
              
              part2_dev_a1_each1 = K_gras_data1 * (1-yita1_l) + K_gras_data2 * (1-yita1_r);
              part2_dev_a1_each2 = K_gras_data1 * (0-yita1_l) + K_gras_data2 * (0-yita1_r);
              part2_dev_a1_each3 = K_gras_data1 * (0-yita1_l) + K_gras_data2 * (0-yita1_r);
              
              dev_a1_each = part1_dev_a_each1 * part2_dev_a1_each1 + part1_dev_a_each2 * part2_dev_a1_each2 + part1_dev_a_each3 * part2_dev_a1_each3;
              % step2, for a2
              part2_dev_a2_each1 = K_spd_data1 * (0-yita2_l) + K_spd_data2 * (0-yita2_r);
              part2_dev_a2_each2 = K_spd_data1 * (1-yita2_l) + K_spd_data2 * (1-yita2_r);
              part2_dev_a2_each3 = K_spd_data1 * (0-yita2_l) + K_spd_data2 * (0-yita2_r);
              
              dev_a2_each = part1_dev_a_each1 * part2_dev_a2_each1 + part1_dev_a_each2 * part2_dev_a2_each2 + part1_dev_a_each3 * part2_dev_a2_each3;
              % step3, for a3
              part2_dev_a3_each1 = K_sgm_data1 * (0-yita3_l) + K_sgm_data2 * (0-yita3_r);
              part2_dev_a3_each2 = K_sgm_data1 * (0-yita3_l) + K_sgm_data2 * (0-yita3_r);
              part2_dev_a3_each3 = K_sgm_data1 * (1-yita3_l) + K_sgm_data2 * (1-yita3_r);
              
              dev_a3_each = part1_dev_a_each1 * part2_dev_a3_each1 + part1_dev_a_each2 * part2_dev_a3_each2 + part1_dev_a_each3 * part2_dev_a3_each3;
            %% caculate the inner part of the dev(Rw)/dev(b)
              % step1, for b1
              part2_dev_b1_each1 = (1-yita1_l) + (1-yita1_r);
              part2_dev_b1_each2 = (0-yita1_l) + (0-yita1_r);
              part2_dev_b1_each3 = (0-yita1_l) + (0-yita1_r);
              
              dev_b1_each = part1_dev_a_each1 * part2_dev_b1_each1 + part1_dev_a_each2 * part2_dev_b1_each2 + part1_dev_a_each3 * part2_dev_b1_each3;
              % step2, for b2
              part2_dev_b2_each1 = (0-yita1_l) + (0-yita1_r);
              part2_dev_b2_each2 = (1-yita1_l) + (1-yita1_r);
              part2_dev_b2_each3 = (0-yita1_l) + (0-yita1_r);
              
              dev_b2_each = part1_dev_a_each1 * part2_dev_b2_each1 + part1_dev_a_each2 * part2_dev_b2_each2 + part1_dev_a_each3 * part2_dev_b2_each3;
              % step3, for b3
              part2_dev_b3_each1 = (0-yita1_l) + (0-yita1_r);
              part2_dev_b3_each2 = (0-yita1_l) + (0-yita1_r);
              part2_dev_b3_each3 = (1-yita1_l) + (1-yita1_r);
              
              dev_b3_each = part1_dev_a_each1 * part2_dev_b3_each1 + part1_dev_a_each2 * part2_dev_b3_each2 + part1_dev_a_each3 * part2_dev_b3_each3;
            
            %% caculate each a's and b's derivate in witnin class
              % step1, for each a
              a1_dev_sum = a1_dev_sum + dev_a1_each; 
              a2_dev_sum = a2_dev_sum + dev_a2_each; 
              a3_dev_sum = a3_dev_sum + dev_a3_each; 
              % step2, for each b
              b1_dev_sum = b1_dev_sum + dev_b1_each;
              b2_dev_sum = b2_dev_sum + dev_b2_each;
              b3_dev_sum = b3_dev_sum + dev_b3_each;
            %% caculate each model's intra-class scatter matrix
              Sw_temp_gras = yita1_l * (K_gras_data1-K_gras_data2) * (K_gras_data1-K_gras_data2)' * yita1_r;
              Sw_temp_spd = yita2_l * (K_spd_data1-K_spd_data2) * (K_spd_data1-K_spd_data2)' * yita2_r;
              Sw_temp_sgm = yita3_l * (K_sgm_data1-K_sgm_data2) * (K_sgm_data1-K_sgm_data2)' * yita3_r;
              Sw_temp = Sw_temp_gras + Sw_temp_spd + Sw_temp_sgm;
              Sw = Sw+Sw_temp;
              Nw = Nw+1; % number of within-class computing pairs
          end
      end
  end
  
  for j=1:num_class
      num_eachclass=find(Train_lables==j); 
      num_difclass=find(Train_lables~=j); 
      for k=1:length(num_eachclass)
          K_gras_data1 = k_Gras(:,num_eachclass(k));
          K_spd_data1 = k_Spd(:,num_eachclass(k));
          K_sgm_data1 = k_Sgm(:,num_eachclass(k));
         %% gating function computing-->denominator-->left side 
          temp_gating_func_sum_left_inter = exp(a1'*K_gras_data1+b1) + exp(a2'*K_spd_data1+b2) + exp(a3'*K_sgm_data1+b3);
         %% caculate each model's weight of left side
          yita1_l_inter = exp(a1'*K_gras_data1+b1) / temp_gating_func_sum_left_inter;
          yita2_l_inter = exp(a2'*K_spd_data1+b2)  / temp_gating_func_sum_left_inter;
          yita3_l_inter = exp(a3'*K_sgm_data1+b3)  /  temp_gating_func_sum_left_inter;
          for m=1:length(num_difclass)
              K_gras_data2 = k_Gras(:,num_difclass(m));
              K_spd_data2 = k_Spd(:,num_difclass(m));
              K_sgm_data2 = k_Sgm(:,num_difclass(m));
            %% gating function computing-->molecule-->right side 
              temp_gating_func_sum_right_inter = exp(a1'*K_gras_data2+b1) + exp(a2'*K_spd_data2+b2) + exp(a3'*K_sgm_data2+b3);
            %% caculate each model's weight of right side
              yita1_r_inter = exp(a1'*K_gras_data2+b1) / temp_gating_func_sum_right_inter;
              yita2_r_inter = exp(a2'*K_spd_data2+b2)  / temp_gating_func_sum_right_inter;
              yita3_r_inter = exp(a3'*K_sgm_data2+b3)  / temp_gating_func_sum_right_inter;
            %% caculate the inner part of the dev(Rw)/dev(a)
              % step1, for a1
              part1_dev_a_each1 = yita1_l_inter * (K_gras_data1-K_gras_data2) * (K_gras_data1-K_gras_data2)' * yita1_r_inter ;
              part1_dev_a_each2 = yita2_l_inter * (K_spd_data1-K_spd_data2) * (K_spd_data1-K_spd_data2)' * yita2_r_inter ;
              part1_dev_a_each3 = yita3_l_inter * (K_sgm_data1-K_sgm_data2) * (K_sgm_data1-K_sgm_data2)' * yita3_r_inter ;
              
              part2_dev_a1_each1 = K_gras_data1 * (1-yita1_l_inter) + K_gras_data2 * (1-yita1_r_inter);
              part2_dev_a1_each2 = K_gras_data1 * (0-yita1_l_inter) + K_gras_data2 * (0-yita1_r_inter);
              part2_dev_a1_each3 = K_gras_data1 * (0-yita1_l_inter) + K_gras_data2 * (0-yita1_r_inter);
              
              dev_a1_each = part1_dev_a_each1 * part2_dev_a1_each1 + part1_dev_a_each2 * part2_dev_a1_each2 + part1_dev_a_each3 * part2_dev_a1_each3;
              % step2, for a2
              part2_dev_a2_each1 = K_spd_data1 * (0-yita2_l_inter) + K_spd_data2 * (0-yita2_r_inter);
              part2_dev_a2_each2 = K_spd_data1 * (1-yita2_l_inter) + K_spd_data2 * (1-yita2_r_inter);
              part2_dev_a2_each3 = K_spd_data1 * (0-yita2_l_inter) + K_spd_data2 * (0-yita2_r_inter);
              
              dev_a2_each = part1_dev_a_each1 * part2_dev_a2_each1 + part1_dev_a_each2 * part2_dev_a2_each2 + part1_dev_a_each3 * part2_dev_a2_each3;
              % step3, for a3
              part2_dev_a3_each1 = K_sgm_data1 * (0-yita3_l_inter) + K_sgm_data2 * (0-yita3_r_inter);
              part2_dev_a3_each2 = K_sgm_data1 * (0-yita3_l_inter) + K_sgm_data2 * (0-yita3_r_inter);
              part2_dev_a3_each3 = K_sgm_data1 * (1-yita3_l_inter) + K_sgm_data2 * (1-yita3_r_inter);
              
              dev_a3_each = part1_dev_a_each1 * part2_dev_a3_each1 + part1_dev_a_each2 * part2_dev_a3_each2 + part1_dev_a_each3 * part2_dev_a3_each3;
            %% caculate the inner part of the dev(Rw)/dev(b)
              % step1, for b1
              part2_dev_b1_each1 = (1-yita1_l_inter) + (1-yita1_r_inter);
              part2_dev_b1_each2 = (0-yita1_l_inter) + (0-yita1_r_inter);
              part2_dev_b1_each3 = (0-yita1_l_inter) + (0-yita1_r_inter);
              
              dev_b1_each = part1_dev_a_each1 * part2_dev_b1_each1 + part1_dev_a_each2 * part2_dev_b1_each2 + part1_dev_a_each3 * part2_dev_b1_each3;
              % step2, for b2
              part2_dev_b2_each1 = (0-yita2_l_inter) + (0-yita2_r_inter);
              part2_dev_b2_each2 = (1-yita2_l_inter) + (1-yita2_r_inter);
              part2_dev_b2_each3 = (0-yita2_l_inter) + (0-yita2_r_inter);
              
              dev_b2_each = part1_dev_a_each1 * part2_dev_b2_each1 + part1_dev_a_each2 * part2_dev_b2_each2 + part1_dev_a_each3 * part2_dev_b2_each3;
              % step3, for b3
              part2_dev_b3_each1 = (0-yita3_l_inter) + (0-yita3_r_inter);
              part2_dev_b3_each2 = (0-yita3_l_inter) + (0-yita3_r_inter);
              part2_dev_b3_each3 = (1-yita3_l_inter) + (1-yita3_r_inter);
              
              dev_b3_each = part1_dev_a_each1 * part2_dev_b3_each1 + part1_dev_a_each2 * part2_dev_b3_each2 + part1_dev_a_each3 * part2_dev_b3_each3;
            %% caculate each a's and b's derivate in witnin class
              % step1, for each a
              a1_dev_sum_inter = a1_dev_sum_inter + dev_a1_each; 
              a2_dev_sum_inter = a2_dev_sum_inter + dev_a2_each; 
              a3_dev_sum_inter = a3_dev_sum_inter + dev_a3_each; 
              % step2, for each b
              b1_dev_sum_inter = b1_dev_sum_inter + dev_b1_each;
              b2_dev_sum_inter = b2_dev_sum_inter + dev_b2_each;
              b3_dev_sum_inter = b3_dev_sum_inter + dev_b3_each;
            %% caculate each model's inter-class scatter matrix
              Sb_temp_gras = yita1_l_inter * (K_gras_data1-K_gras_data2) * (K_gras_data1-K_gras_data2)' * yita1_r_inter;
              Sb_temp_spd = yita2_l_inter * (K_spd_data1-K_spd_data2) * (K_spd_data1-K_spd_data2)' * yita2_r_inter;
              Sb_temp_sgm = yita3_l_inter * (K_sgm_data1-K_sgm_data2) * (K_sgm_data1-K_sgm_data2)' * yita3_r_inter;
              Sb_temp = Sb_temp_gras + Sb_temp_spd + Sb_temp_sgm;
              Sb = Sb + Sb_temp;
              Nb = Nb + 1; % number of between-class computing pairs
          end
      end
  end
  
  Sw_final = Sw/(Nw); % Eq.(9),400 * 400
  Sb_final = Sb/(Nb); % Eq.(10),400 * 400
  
 %% iterative optimization

  St_final = Sw_final + Sb_final; % step1
  [ W1, ~, W2 ] = svd(St_final); % step2, in Y, the positive sigular's number is 40
  Sb_final_w = W1' * Sb_final * W2; % in order to construct Equ.11
  St_final_w = W1' * St_final * W2;
  
  for k = 1 : itera_innner
      
      Lmd_k = trace(V0' * Sb_final_w * V0) / trace(V0' * St_final_w * V0);
      temp = Sb_final_w - Lmd_k * St_final_w; % Eq.13
      
      [Object_V , Object_E] = eig(temp); % max
      E_unsort = diag(Object_E);
      [~ , index] = sort(E_unsort,'descend'); % 
      V_sort = Object_V(:,index); %
      V = V_sort(:,1:d); % get V's value Eq.14
      
      St_final_v = V * V' * St_final_w * V * V';
      [ V_kk_l, ~, ~ ] = svd(St_final_v);
      V0 = V_kk_l(:,1:d);
      V_compute{k} = V0;
      
  end
  
  U = W1 * V0; % the final transformation matrix achieved by iterative procedure
  % get each parameter's derivative
  value_Sw = trace( U' * Sw_final * U ); 
  value_Sb = trace( U' * Sb_final * U );
  
  dev_a1 = ((U * U') * (a1_dev_sum_inter * value_Sw - value_Sb * a1_dev_sum)) / ((value_Sw+value_Sb)^2); % derivative of loss with regard to a1
  dev_a2 = ((U * U') * (a2_dev_sum_inter * value_Sw - value_Sb * a2_dev_sum)) / ((value_Sw+value_Sb)^2); % derivative of loss with regard to a2
  dev_a3 = ((U * U') * (a3_dev_sum_inter * value_Sw - value_Sb * a3_dev_sum)) / ((value_Sw+value_Sb)^2); % derivative of loss with regard to a3
  
  dev_b1_whole = ((U * U') * (b1_dev_sum_inter * value_Sw - value_Sb * b1_dev_sum)) / ((value_Sw+value_Sb)^2); % derivative of loss with regard to b1
  dev_b1 = sum(diag(dev_b1_whole));
  dev_b2_whole = ((U * U') * (b2_dev_sum_inter * value_Sw - value_Sb * b2_dev_sum)) / ((value_Sw+value_Sb)^2); % derivative of loss with regard to b2
  dev_b2 = sum(diag(dev_b2_whole));
  dev_b3_whole = ((U * U') * (b3_dev_sum_inter * value_Sw - value_Sb * b3_dev_sum)) / ((value_Sw+value_Sb)^2); % derivative of loss with regard to b3
  dev_b3 = sum(diag(dev_b3_whole));
  
  % gradient ascent
  a_1(:,i+1) = a_1(:,i) + alpha * dev_a1;
  a1 = a_1(:,i+1);
  
  a_2(:,i+1) = a_2(:,i) + alpha * dev_a2;
  a2 = a_2(:,i+1);
%   
  a_3(:,i+1) = a_3(:,i) + alpha * dev_a3;
  a3 = a_3(:,i+1);
  
  b_1(:,i+1) = b_1(:,i) + alpha * dev_b1;
  b1 = b_1(:,i+1);
  
  b_2(:,i+1) = b_2(:,i) + alpha * dev_b2;
  b2 = b_2(:,i+1);
%   
  b_3(:,i+1) = b_3(:,i) + alpha * dev_b3;
  b3 = b_3(:,i+1);
  
 %% compute the cost
  Cost(i) = det(U'*Sb_final*U)/det(U'*Sw_final*U);
  fprintf(' iter\t            cost val\t \n    ');
  fprintf('%5d\t \n%+.16e\t \n', i, Cost(:,1:i));
  
 %% store each parameter
  a_all(:,1) = a1;
  a_all(:,2) = a2;
  a_all(:,3) = a3;
  b_all(:,1) = b1;
  b_all(:,2) = b2;
  b_all(:,3) = b3;
  
end

end



