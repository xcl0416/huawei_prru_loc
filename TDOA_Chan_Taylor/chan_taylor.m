% %% the simulation of TDOA localization algorithm 
clear all; 
clc; 
%定义四个参与基站的坐标位置

BS3 = [5, 9];
BS1=[0,0];

BS2=[1,6];
BS4=[1,8];
BS5=[9,1];
BS6 = [3, 2];


sample = 100;
RM1 = zeros(1, 5);
RM2 = zeros(1, 5);
RM = zeros(1, 5);

error = 0;
for i= 1:1000
    BS1_ = BS1 + (10)*randn(size(BS1)) * 0.3;
    error = error + norm(BS1_- BS1);
end
error = error / 1000
RMSE1 = ones(1, 5)
tic
for sp=1:sample

    BS1 = round(random('uniform', 0, 1, 1, 2), 1);
    BS3 = [round(random('uniform', 0, 1, 1, 1), 1), round(random('uniform', 19, 20, 1, 1), 1)];
    BS2 = [round(random('uniform', 19, 20, 1, 1), 1), round(random('uniform', 0, 1, 1, 1), 1)];
    BS4 = round(random('uniform', 19, 20, 1, 2), 1);
%     BS5 = round(random('uniform', 0, 1, 1, 2), 1);
%     BS6 = round(random('uniform', 0, 1, 1, 2), 1);
    BS5 = [round(random('uniform', 9, 11, 1, 1), 1), round(random('uniform', 9, 11, 1, 1), 1)];
    BS6 = [round(random('uniform', 9, 11, 1, 1), 1), round(random('uniform', 19, 20, 1, 1), 1)];

    BS7 = round(random('uniform', 0, 20, 1, 2), 1);

%     BS1=[0,0];
%     BS2=[1,19];
%     BS3 = [19, 19];
%     BS4=[19,1];
%     BS5=[9,8];
%     BS6 = [3, 2];

    %BS5=[600,500];
    %移动台MS的初始估计位置
    MS=[0,0]; 
    MS = round(random('uniform', 0, 20, 1, 2), 1);
%     MS=[10,12]; 

%     std_var=[1e-2,5e-2,1e-1,5e-1,1]; %范围
    std_var=[1,2,3,4,5]; %范围

    %A=[BS1;BS2;BS3;BS4]; %矩阵A包含4个初始坐标
    A=[BS1;BS2;BS3;BS4;BS5;BS6];
    

    n_count1 = 0;
    n_count2 = 0;
    n_count3 = 0;
    n_count4 = 0;
    n_count5 = 0;
    n_count6 = 0;

    number=50;
    
    for j=1:1 %循环

         error1=0;%初始误差置为0
         error2=0; %初始误差置为0
         std_var1=std_var(j);%令std_var1等于当前数组的值
         error_final = 0;
         for i=1:number %多次循环

             tmp_c = 1000;
              %r1=A-ones(4,1)*MS; 
             A_noise = A + (10)*randn(size(A)) * 0.3;
             A_noise(1,:) = A(1,:);
             A_noise(2,:) = A(2,:);
%              A_noise(5,:) = A(5,:);

             AA_noise = A_noise;
             AA = A;
             num_u = size(A, 1);
             err = [];
             for y = 1:size(A, 1)
                  AA_noise = A_noise;
                  AA = A;
                  AA(1, :) = A(y, :);
                  AA(y, :) = A(1, :);
                  AA_noise(1, :) = A_noise(y, :);
                  AA_noise(y, :) = A_noise(1, :);
                  theta2_arr_tmp = [];
                  theta1_arr_tmp = [];
                  dif = 0;
                  for ind1 = 2:num_u - 3
                    for ind2 = ind1 + 1:num_u - 2
                        for ind3 = ind2 + 1:num_u - 1
                            for ind4 = ind3 + 1:num_u
                                index_ue = [1, ind1, ind2, ind3, ind4];
                                AA_tmp = AA(index_ue,:);
                                AA_tmp_noise = AA_noise(index_ue,:);

                                r1=AA_tmp-ones(size(AA_tmp, 1),1)*MS;
                                r2=(sum(r1.^2,2)).^(1/2); 
                                %r=r2(2:end,:)-ones(3,1)*r2(1,:)+std_var1*randn(3,1); %表示从[2,i]开始MS与基站i和基站1的距离差
                                r=r2(2:end,:)-ones(size(AA_tmp, 1) - 1,1)*r2(1,:)+std_var1*randn(size(AA_tmp, 1) - 1,1) * 0.3;
                                sigma=std_var1^2; 
%                                 theta1=TDOACHAN(AA_tmp_noise,r,sigma); % 调用TDOACHAN函数
                                theta2=TDOATaylor(AA_tmp_noise,r,sigma);
                                r1_=AA_tmp-ones(size(AA_tmp, 1),1)*theta2;
                                r2_=(sum(r1_.^2,2)).^(1/2); 
                                %r=r2(2:end,:)-ones(3,1)*r2(1,:)+std_var1*randn(3,1); %表示从[2,i]开始MS与基站i和基站1的距离差

                                r_=r2_(2:end,:)-ones(size(AA_tmp, 1) - 1,1)*r2_(1,:)+std_var1*randn(size(AA_tmp, 1) - 1,1) * 0.3;

    %                             AAA =  (r - r_)
                                dif = dif + sum(abs(r-r_));

                                theta2_arr_tmp = [theta2_arr_tmp; theta2];
%                                 theta1_arr_tmp = [theta1_arr_tmp; theta1];





                            end
                        end
                    end
                  end
    %               theta2_arr_tmp
    %                   aa = theta2_arr_tmp(:, 1)
    %               tmp_var = var(theta2_arr_tmp(:, 1)) + var(theta2_arr_tmp(:, 2));
                  tmp_var = dif;
                  if tmp_var <= tmp_c
                      tmp_c = tmp_var;
                      r1=AA-ones(size(AA, 1),1)*MS;
                      r2=(sum(r1.^2,2)).^(1/2); 
                      %r=r2(2:end,:)-ones(3,1)*r2(1,:)+std_var1*randn(3,1); %表示从[2,i]开始MS与基站i和基站1的距离差
                      r=r2(2:end,:)-ones(size(AA, 1) - 1,1)*r2(1,:)+std_var1*randn(size(AA, 1) - 1,1) * 0.3;
                      sigma=std_var1^2; 
    %                       theta1=TDOACHAN(AA_noise,r,sigma);
    %                       theta2=TDOATaylor(AA_noise,r,sigma);
                      theta2_final5 = mean(theta2_arr_tmp, 1);
                      theta2_final6 = TDOATaylor(AA_noise,r,sigma);
%                       theta1_final5 = mean(theta1_arr_tmp, 1);
%                       theta1_final6 = TDOACHAN(AA_noise,r,sigma);

                      err = [err, [norm(MS-theta2_final6); norm(MS-theta2_final5)]];

                      best1 = y;
                      A_noise_ = A_noise;

                      A_noise_(1,:) = A(1,:);
                      A_noise_(2,:) = A(2,:);
                      r1=A-ones(size(A, 1),1)*MS;
                      r2=(sum(r1.^2,2)).^(1/2); 
                      %r=r2(2:end,:)-ones(3,1)*r2(1,:)+std_var1*randn(3,1); %表示从[2,i]开始MS与基站i和基站1的距离差
                      r=r2(2:end,:)-ones(size(A, 1) - 1,1)*r2(1,:)+std_var1*randn(size(A, 1) - 1,1) * 0.3;
                      sigma=std_var1^2; 
    %                       theta1=TDOACHAN(AA_noise,r,sigma);
    %                       theta2=TDOATaylor(AA_noise,r,sigma);
%                       theta2_final5 = mean(theta2_arr_tmp, 1);
                      theta2_final_nochoice = TDOATaylor(A_noise_,r,sigma);
                  end

            end
            theta_final = (theta2_final6 + theta2_final5) / 2;
            error_final = error_final + norm(MS-theta_final);
            error1 = error1 + norm(MS-theta2_final_nochoice);
%             a = 1;
        if best1 == 1
            n_count1 = n_count1 + 1;
        elseif best1 == 2
            n_count2 = n_count2 + 1;
        elseif best1 == 3
            n_count3 = n_count3 + 1;
        elseif best1 == 4
            n_count4 = n_count4 + 1;
        elseif best1 == 5
            n_count5 = n_count5 + 1;
        elseif best1 == 6
            n_count6 = n_count6 + 1;
        end
        end
        RMSE(j) = error_final / number;
        RMSE1(j) = error1 / number;

        toc
%               A_=A + (3)*randn(size(A)) * 0.3;
    %               A_(1,:) = A(1,:);
    %               A_(2,:) = A(2,:);



%               r1=A-ones(size(A, 1),1)*MS;
%               r2=(sum(r1.^2,2)).^(1/2); 
%               %r=r2(2:end,:)-ones(3,1)*r2(1,:)+std_var1*randn(3,1); %表示从[2,i]开始MS与基站i和基站1的距离差
%               r=r2(2:end,:)-ones(size(A, 1) - 1,1)*r2(1,:)+std_var1*randn(size(A, 1) - 1,1) * 0.3;
%               sigma=std_var1^2; 
%               theta1=TDOACHAN(A_,r,sigma); % 调用TDOACHAN函数
%               theta2=TDOATaylor(A_,r,sigma); %调用TDOATalor函数
%     %           a = norm(MS-theta1)^2
%               theta1 = real(theta1);
%               error1=error1+norm(MS-theta1); %移动台MS估计位置与计算的到的距离的平方
%               error2=error2+norm(MS-theta2); %移动台MS估计位置与计算的到的距离的平方
%         end 
%           RMSE1(j)=(error1/number) %均方根误差
%           RMSE2(j)=(error2/number) %均方根误差
    end
    RM = (RM + RMSE);

    RM1 = (RM1 + RMSE1);
%     RM2 = (RM2 + RMSE2);
end
RM1 = RM1 / sample
% RM2 = RM2 / sample
RM = RM / sample

%     % plot
% semilogx(std_var,RMSE1,'-O',std_var,RMSE2,'-s')% x轴取对数，X轴范围是1e-2到1,Y轴的范围是变动的
% xlabel('The standard deviation of measurement noise (m)'); 
% ylabel('RMSE'); 
% legend('TDOA-CHAN','TDOA-Taylor');
% scatter(A(:,1),A(:,2),'r')
% hold on
% scatter(MS(1), MS(2))
% axis([0 20, 0, 20])

