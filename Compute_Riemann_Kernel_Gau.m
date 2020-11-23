function [outKernel, sigma_V]=Compute_Riemann_Kernel_Gau(CY1,CY2)
if(isempty(CY2))
    CY2 = CY1;
end
number_sets1=length(CY1);
number_sets2=length(CY2);

outKernel=zeros(number_sets1,number_sets2);
for tmpC1=1:number_sets1
    Y1=CY1{tmpC1};
    Y1_T = Y1.T;
    Y1_W = Y1.W;
    for tmpC2=1:number_sets2
        Y2=CY2{tmpC2};
        Y2_T = Y2.T;      
        Y2_W = Y2.W;
        temp = 0;
        for ii = 1 : length(Y1_T)
            Y1_T_ii = Y1_T{ii};
            Y1_T_ii = Y1_T_ii(:);
            for jj = 1 : length(Y2_T)
                Y2_T_jj = Y2_T{jj};
                Y2_T_jj = Y2_T_jj(:);
                temp = temp+Y1_W(ii)*Y2_W(jj)*Y1_T_ii'*Y2_T_jj;%trace(Y1_T_ii*Y2_T_jj);
            end
        end
        outKernel(tmpC1,tmpC2)=temp;
    end
end
sigma_V = 0;
