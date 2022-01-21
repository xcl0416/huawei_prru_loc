clc;clear;close all;
tic
ti = 100;
data_ue_error = ones(ti,1);
data_bs_error = ones(ti,1);
BS3_arr_final = ones(ti,2);
BS4_arr_final = ones(ti,2);
% a = data_ue_error(11)
for y =1:1:ti
    BS1_true = [1,1];BS2_true = [19,1];BS3_true = [1,19];BS4_true = [19,19];
    UE1_true = [10,5];UE2_true = [4,9];UE3_true = [7,3];
    UE1_SET = [];UE2_SET = [];UE3_SET = [];UE4_SET = [];
    UE_num = 3;
    std = 2;

    for UE1_x = 0:20
        for UE1_y = 0:20
            UE_est = [UE1_x,UE1_y];
            d12 = get_distace(BS1_true,UE_est)-get_distace(BS2_true,UE_est);
            d12_true = get_distace(BS1_true,UE1_true)-get_distace(BS2_true,UE1_true);
            if abs(d12-d12_true)<2^(1/2) / 8
                UE1_SET = [UE1_SET;UE_est];
            end
        end
    end
    for UE2_x = 0:20
        for UE2_y = 0:20
            UE_est = [UE2_x,UE2_y];
            d12 = get_distace(BS1_true,UE_est)-get_distace(BS2_true,UE_est);
            d12_true = get_distace(BS1_true,UE2_true)-get_distace(BS2_true,UE2_true);
            if abs(d12-d12_true)<2^(1/2) / 8
                UE2_SET = [UE2_SET;UE_est];
            end
        end
    end

    for UE3_x = 0:20
        for UE3_y = 0:20
            UE_est = [UE3_x,UE3_y];
            d12 = get_distace(BS1_true,UE_est)-get_distace(BS2_true,UE_est);
            d12_true = get_distace(BS1_true,UE3_true)-get_distace(BS2_true,UE3_true);
            if abs(d12-d12_true)<2^(1/2) / 8
                UE3_SET = [UE3_SET;UE_est];
            end
        end
    end

    error3_min = 1000000;
    error4_min = 1000000;
    error_34 = 1000000;
    % h = waitbar(0,'running');
    for i =1:length(UE1_SET)

        for j =1:length(UE2_SET)
            for k =1:length(UE3_SET)

                UE1_est = UE1_SET(i,:);
                UE2_est = UE2_SET(j,:);
                UE3_est = UE3_SET(k,:);
    %             UE4_est = UE4_SET(f,:);
%                 UE1_est = [10,5];
%                 UE2_est = [4,9];
%                 UE3_est = [7,3];
    %             UE4_est = [19,15];
                R1_3 = get_distace(BS1_true,UE1_est)-...
                    (get_distace(BS1_true,UE1_true)-get_distace(BS3_true,UE1_true))+(std*randn(1)*0.3);
                R1_4 = get_distace(BS1_true,UE1_est)-...
                    (get_distace(BS1_true,UE1_true)-get_distace(BS4_true,UE1_true))+(std*randn(1)*0.3);
                R2_3 = get_distace(BS1_true,UE2_est)-...
                    (get_distace(BS1_true,UE2_true)-get_distace(BS3_true,UE2_true))+(std*randn(1)*0.3);
                R2_4 = get_distace(BS1_true,UE2_est)-...
                    (get_distace(BS1_true,UE2_true)-get_distace(BS4_true,UE2_true))+(std*randn(1)*0.3);
                R3_3 = get_distace(BS1_true,UE3_est)-...
                    (get_distace(BS1_true,UE3_true)-get_distace(BS3_true,UE3_true))+(std*randn(1)*0.3);
                R3_4 = get_distace(BS1_true,UE3_est)-...
                    (get_distace(BS1_true,UE3_true)-get_distace(BS4_true,UE3_true))+(std*randn(1)*0.3); 

                R_3 = [R1_3; R2_3; R3_3];
                UE_est_123 = [UE1_est;UE2_est;UE3_est];
    %                 BS3_est_x = [];
    %                 BS3_est_y = [];
                err3_arr = [];
                index3_arr = [];
    %             BS3_arr = [-100,-200,-300,-400;-500,-600,-700,-800;-900,-1000,-1100,-1200];
                BS3_list = linspace(100, 100 * UE_num * (UE_num - 1) / 2 * 4, UE_num * (UE_num - 1) / 2 * 4);
                BS3_arr = reshape(BS3_list, UE_num * (UE_num - 1) / 2, 4);
                n3 = 0;

                for a = 1:1:UE_num
                        for b = a+1:1:UE_num

                            n3 = n3 + 1;
                            dis_ab = get_distace(UE_est_123(a,:), UE_est_123(b,:));
                            if R_3(a) + R_3(b) >= dis_ab && dis_ab >= abs(R_3(a) - R_3(b))
    %                             [B3_x, B3_y] = ls(UE_est_123(a, 1), UE_est_123(a, 2), R_3(a), UE_est_123(b, 1), UE_est_123(b, 2), R_3(b));
    %                             BS3_est_x = [BS3_est_x; sum(B3_x) / 2];
    %                             BS3_est_y = [BS3_est_y; sum(B3_y) / 2];
                                BS3_est_x_temp = (((R_3(a)^2 - R_3(b)^2 + dis_ab^2) / (2 * dis_ab)) / dis_ab) * (UE_est_123(b, 1) - UE_est_123(a, 1)) + UE_est_123(a, 1);
                                BS3_est_y_temp = (((R_3(a)^2 - R_3(b)^2 + dis_ab^2) / (2 * dis_ab)) / dis_ab) * (UE_est_123(b, 2) - UE_est_123(a, 2)) + UE_est_123(a, 2);
    %                             BS3_est_x = [BS3_est_x; BS3_est_x_temp];
    %                             BS3_est_y = [BS3_est_y; BS3_est_y_temp];
                                x1 = BS3_est_x_temp - ((R_3(a)^2 - ((R_3(a)^2 - R_3(b)^2 + dis_ab^2) / (2 * dis_ab))^2) / (1 + ((UE_est_123(b, 1) - UE_est_123(a, 1)) ^2 / (UE_est_123(b, 2) - UE_est_123(a, 2))^2)))^(1/2);
                                y1 = BS3_est_y_temp - ((UE_est_123(b, 1) - UE_est_123(a, 1)) / (UE_est_123(b, 2) - UE_est_123(a, 2))) * (x1 - BS3_est_x_temp);
                                x2 = BS3_est_x_temp + ((R_3(a)^2 - ((R_3(a)^2 - R_3(b)^2 + dis_ab^2) / (2 * dis_ab))^2) / (1 + ((UE_est_123(b, 1) - UE_est_123(a, 1)) ^2 / (UE_est_123(b, 2) - UE_est_123(a, 2))^2)))^(1/2);
                                y2 = BS3_est_y_temp - ((UE_est_123(b, 1) - UE_est_123(a, 1)) / (UE_est_123(b, 2) - UE_est_123(a, 2))) * (x2 - BS3_est_x_temp);

    %                             a = 1
                                BS3_arr(n3, 1) = x1;
                                BS3_arr(n3, 2) = y1;
                                BS3_arr(n3, 3) = x2;
                                BS3_arr(n3, 4) = y2;
    % %                             a = 1


                            end
                            if R_3(a) + R_3(b) < dis_ab
                                BS3_est_x_temp = (((dis_ab + R_3(a) - R_3(b)) / (2 * dis_ab))) * (UE_est_123(b, 1) - UE_est_123(a, 1)) + UE_est_123(a, 1);
                                BS3_est_y_temp = (((dis_ab + R_3(a) - R_3(b)) / (2 * dis_ab))) * (UE_est_123(b, 2) - UE_est_123(a, 2)) + UE_est_123(a, 2);
    %                             BS3_est_x = [BS3_est_x; BS3_est_x_temp];
    %                             BS3_est_y = [BS3_est_y; BS3_est_y_temp];
                                BS3_arr(n3, 1) = BS3_est_x_temp;
                                BS3_arr(n3, 2) = BS3_est_y_temp;
                            end
                            if  dis_ab < abs(R_3(a) - R_3(b))
    %                             if R_3(a) > R_3(b)
                                BS3_est_x_temp = (((R_3(a) + R_3(b) + dis_ab) / (2 * dis_ab))) * (UE_est_123(b, 1) - UE_est_123(a, 1)) + UE_est_123(a, 1);
                                BS3_est_y_temp = (((R_3(a) + R_3(b) + dis_ab) / (2 * dis_ab))) * (UE_est_123(b, 2) - UE_est_123(a, 2)) + UE_est_123(a, 2);
    %                             BS3_est_x = [BS3_est_x; BS3_est_x_temp];
    %                             BS3_est_y = [BS3_est_y; BS3_est_y_temp]; 
                                BS3_arr(n3, 1) = BS3_est_x_temp;
                                BS3_arr(n3, 2) = BS3_est_y_temp;

                            end

                        end
                end
    %                 for p=1:1:UE_num * (UE_num - 1) / 2 - 1
    %                     for y=p+1:1:UE_num * (UE_num - 1) / 2
    %                         err3_arr = [err3_arr; norm(BS3_arr(p, 1:2) - BS3_arr(y,1:2))];
    %                         index3_arr = [index3_arr; [p, 1, y, 1]];
    %                         err3_arr = [err3_arr; norm(BS3_arr(p, 1:2) - BS3_arr(y,3:4))];
    %                         index3_arr = [index3_arr; [p, 1, y, 3]];
    %                         err3_arr = [err3_arr; norm(BS3_arr(p, 3:4) - BS3_arr(y,1:2))];
    %                         index3_arr = [index3_arr; [p, 3, y, 1]];
    %                         err3_arr = [err3_arr; norm(BS3_arr(p, 3:4) - BS3_arr(y,3:4))];
    %                         index3_arr = [index3_arr; [p, 3, y, 3]];
    %                     end
    %                 end
                error3_min = 1000000;

                for p1 = 1:1:2
                    if BS3_arr(1, p1 * 2) > -3 && BS3_arr(1, p1 * 2) < 23 && BS3_arr(1, p1 * 2 - 1) > -3 && BS3_arr(1, p1 * 2 - 1) <23
                        for p2 = 1:1:2
                            if BS3_arr(2, p2 * 2) > -3 && BS3_arr(2, p2 * 2) < 23 && BS3_arr(2, p2 * 2 - 1) > -3 && BS3_arr(2, p2 * 2 - 1) <23

                                for p3 = 1:1:2
                                    if BS3_arr(3, p3 * 2) > -3 && BS3_arr(3, p3 * 2) < 23 && BS3_arr(3, p3 * 2 - 1) > -3 && BS3_arr(3, p3 * 2 - 1) <23

                %                         arrr = [BS3_arr(1, p1 * 2 - 1),BS3_arr(2,  p2 * 2 - 1), BS3_arr(3,  p3 * 2 - 1)]
                                        error3 = var([BS3_arr(1, p1 * 2 - 1),BS3_arr(2,  p2 * 2 - 1), BS3_arr(3,  p3 * 2 - 1)]) +...
                                            var([BS3_arr(1, p1 * 2),BS3_arr(2,  p2 * 2), BS3_arr(3,  p3 * 2)]);
                                        index3_arr = [p1;p2;p3];
                                        if error3 < error3_min
                                            error3_min = error3;
                %                             err_3_arr = err3_arr;
                                            index_3_arr = index3_arr;
                %                             BS3_best = BS3_arr;
                %                             UE1_best3 = UE1_est;
                %                             UE2_best3 = UE2_est;
                %                             UE3_best3 = UE3_est;
                                        end



                                    end
                                end
                            end
                        end
                    end

                end
    %                 error3 = min(err3_arr);

    %                 if error3 < error3_min
    %                     error3_min = error3;
    %                     err_3_arr = err3_arr;
    %                     index_3_arr = index3_arr;
    %                     BS3_best = BS3_arr;
    %                     UE1_best3 = UE1_est;
    %                     UE2_best3 = UE2_est;
    %                     UE3_best3 = UE3_est;
    % 
    % 
    %                 end


                R_4 = [R1_4; R2_4; R3_4];
    %             UE_est_123 = [UE1_est;UE2_est;UE3_est];
    %                 BS4_est_x = [];
    %                 BS4_est_y = [];
                err4_arr = [];
                index4_arr = [];
    %             BS4_arr = [-100,-200,-300,-400;-500,-600,-700,-800;-900,-1000,-1100,-1200];
                BS4_list = linspace(100, 100 * UE_num * 4, UE_num * (UE_num - 1) / 2 * 4);
                BS4_arr = reshape(BS4_list, UE_num * (UE_num - 1) / 2, 4);
                n4 = 0;
                for a = 1:1:UE_num
                        for b = a+1:1:UE_num

    %                         if a ~= b
                            dis_ab = get_distace(UE_est_123(a,:), UE_est_123(b,:));
                            n4 = n4 + 1 ;
                            if R_4(a) + R_4(b) >= dis_ab && dis_ab >= abs(R_4(a) - R_4(b))
    %                             [B4_x, B4_y] = ls(UE_est_123(a, 1), UE_est_123(a, 2), R_4(a), UE_est_123(b, 1), UE_est_123(b, 2), R_4(b));
    %                             BS4_est_x = [BS4_est_x; sum(B4_x) / 2];
    %                             BS4_est_y = [BS4_est_y; sum(B4_y) / 2];
                                BS4_est_x_temp = (((R_4(a)^2 - R_4(b)^2 + dis_ab^2) / (2 * dis_ab)) / dis_ab) * (UE_est_123(b, 1) - UE_est_123(a, 1)) + UE_est_123(a, 1);
                                BS4_est_y_temp = (((R_4(a)^2 - R_4(b)^2 + dis_ab^2) / (2 * dis_ab)) / dis_ab) * (UE_est_123(b, 2) - UE_est_123(a, 2)) + UE_est_123(a, 2);
    %                             BS4_est_x = [BS4_est_x; BS4_est_x_temp];
    %                             BS4_est_y = [BS4_est_y; BS4_est_y_temp];
                                x1 = BS4_est_x_temp - ((R_4(a)^2 - ((R_4(a)^2 - R_4(b)^2 + dis_ab^2) / (2 * dis_ab))^2) / (1 + ((UE_est_123(b, 1) - UE_est_123(a, 1)) ^2 / (UE_est_123(b, 2) - UE_est_123(a, 2)) ^2)))^(1/2);
                                y1 = BS4_est_y_temp - ((UE_est_123(b, 1) - UE_est_123(a, 1)) / (UE_est_123(b, 2) - UE_est_123(a, 2))) * (x1 - BS4_est_x_temp);
                                x2 = BS4_est_x_temp + ((R_4(a)^2 - ((R_4(a)^2 - R_4(b)^2 + dis_ab^2) / (2 * dis_ab))^2) / (1 + ((UE_est_123(b, 1) - UE_est_123(a, 1)) ^2 / (UE_est_123(b, 2) - UE_est_123(a, 2)) ^2)))^(1/2);
                                y2 = BS4_est_y_temp - ((UE_est_123(b, 1) - UE_est_123(a, 1)) / (UE_est_123(b, 2) - UE_est_123(a, 2))) * (x2 - BS4_est_x_temp);
    %                             a = 1
                                BS4_arr(n4, 1) = x1;
                                BS4_arr(n4, 2) = y1;
                                BS4_arr(n4, 3) = x2;
                                BS4_arr(n4, 4) = y2;

                            end
                            if R_4(a) + R_4(b) < dis_ab
                                BS4_est_x_temp = (((dis_ab + R_4(a) - R_4(b)) / (2 * dis_ab))) * (UE_est_123(b, 1) - UE_est_123(a, 1)) + UE_est_123(a, 1);
                                BS4_est_y_temp = (((dis_ab + R_4(a) - R_4(b)) / (2 * dis_ab))) * (UE_est_123(b, 2) - UE_est_123(a, 2)) + UE_est_123(a, 2);
    %                             BS4_est_x = [BS4_est_x; BS4_est_x_temp];
    %                             BS4_est_y = [BS4_est_y; BS4_est_y_temp];
                                BS4_arr(n4, 1) = BS4_est_x_temp;
                                BS4_arr(n4, 2) = BS4_est_y_temp;
                            end
                            if  dis_ab < abs(R_4(a) - R_4(b))
    %                             if R_3(a) > R_3(b)
                                BS4_est_x_temp = (((R_4(a) + R_4(b) + dis_ab) / (2 * dis_ab))) * (UE_est_123(b, 1) - UE_est_123(a, 1)) + UE_est_123(a, 1);
                                BS4_est_y_temp = (((R_4(a) + R_4(b) + dis_ab) / (2 * dis_ab))) * (UE_est_123(b, 2) - UE_est_123(a, 2)) + UE_est_123(a, 2);
    %                             BS4_est_x = [BS4_est_x; BS4_est_x_temp];
    %                             BS4_est_y = [BS4_est_y; BS4_est_y_temp];
                                BS4_arr(n4, 1) = BS4_est_x_temp;
                                BS4_arr(n4, 2) = BS4_est_y_temp;

                            end
    %                         end
                        end
                end
    %                 for p=1:1:UE_num * (UE_num - 1) / 2 - 1
    %                     for y=p+1:1:UE_num * (UE_num - 1) / 2
    %                         err4_arr = [err4_arr; norm(BS4_arr(p, 1:2) - BS4_arr(y,1:2))];
    %                         index4_arr = [index4_arr; [p, 1, y, 1]];
    %                         err4_arr = [err4_arr; norm(BS4_arr(p, 1:2) - BS4_arr(y,3:4))];
    %                         index4_arr = [index4_arr; [p, 1, y, 3]];
    %                         err4_arr = [err4_arr; norm(BS4_arr(p, 3:4) - BS4_arr(y,1:2))];
    %                         index4_arr = [index4_arr; [p, 3, y, 1]];
    %                         err4_arr = [err4_arr; norm(BS4_arr(p, 3:4) - BS4_arr(y,3:4))];
    %                         index4_arr = [index4_arr; [p, 3, y, 3]];
    %                     end
    %                 end

    %             error4 = var(BS4_est_x) + var(BS4_est_y);
    %                 error4 = min(err4_arr);
    %             [error4, ind] = min(err4_arr);
                error4_min = 1000000;

                for p1 = 1:1:2
                    if BS4_arr(1, p1 * 2) > -3 && BS4_arr(1, p1 * 2) < 23 && BS4_arr(1, p1 * 2 - 1) > -3 && BS4_arr(1, p1 * 2 - 1) <23

                        for p2 = 1:1:2
                            if BS4_arr(2, p2 * 2) > -3 && BS4_arr(2, p2 * 2) < 23 && BS4_arr(2, p2 * 2 - 1) > -3 && BS4_arr(2, p2 * 2 - 1) <23

                                for p3 = 1:1:2
                                    if BS4_arr(3, p3 * 2) > -3 && BS4_arr(3, p3 * 2) < 23 && BS4_arr(3, p3 * 2 - 1) > -3 && BS4_arr(3, p3 * 2 - 1) <23


                %                                         arrr = [BS3_arr(1, p1 * 2 - 1),BS3_arr(2,  p2 * 2 - 1), BS3_arr(3,  p3 * 2 - 1),BS3_arr(4,  p4 * 2 - 1),BS3_arr(5,  p5 * 2 - 1),BS3_arr(6,  p6 * 2 - 1)]
                                        error4 = var([BS4_arr(1, p1 * 2 - 1),BS4_arr(2,  p2 * 2 - 1), BS4_arr(3,  p3 * 2 - 1)]) +...
                                            var([BS4_arr(1, p1 * 2),BS4_arr(2,  p2 * 2), BS4_arr(3,  p3 * 2)]);
                                        index4_arr = [p1;p2;p3];
                                        if error4 < error4_min
                                            error4_min = error4;
                %                             err_4_arr = err4_arr;
                                            index_4_arr = index4_arr;
                %                             BS4_best = BS4_arr;
                %                             UE1_best4 = UE1_est;
                %                             UE2_best4 = UE2_est;
                %                             UE3_best4 = UE3_est;





                                        end
                                    end
                                end
                            end
                        end
                    end

                end
                if error3_min + error4_min < error_34
                    error_34 = error3_min + error4_min;
                    err_4_arr = err4_arr;
    %                 index_4_arr = index4_arr;
                    index_4_arr_final = index_4_arr;

                    BS4_best = BS4_arr;
    %                 UE1_best4 = UE1_est;
    %                 UE2_best4 = UE2_est;
    %                 UE3_best4 = UE3_est;
                    err_3_arr = err3_arr;
    %                 index_3_arr = index3_arr;
                    index_3_arr_final = index_3_arr;

                    BS3_best = BS3_arr;
                    UE1_best3 = UE1_est;
                    UE2_best3 = UE2_est;
                    UE3_best3 = UE3_est;

                end


    %                 if error4 < error4_min
    %     %                 [error4, ind3] = min(err4_arr);
    %                     error4_min = error4;
    %                     err_4_arr = err4_arr;
    %                     BS4_best = BS4_arr;
    %                     UE1_best4 = UE1_est;
    %                     UE2_best4 = UE2_est;
    %                     UE3_best4 = UE3_est;
    %                 end




            end
    %         disp(j)
        end
    %     print(i)

    %     waitbar(i/length(UE1_SET))
    end
    % close(h)
    % error_BS = (abs(get_distace(BS3_best,BS3_true))+abs(get_distace(BS4_best,BS4_true)))/2
    % error_UE3 = (abs(get_distace(UE1_best3,UE1_true))...
    %         +abs(get_distace(UE2_best3,UE2_true))...
    %         +abs(get_distace(UE3_best3,UE3_true)))/3
    % error_UE4 = (abs(get_distace(UE1_best4,UE1_true))...
    %         +abs(get_distace(UE2_best4,UE2_true))...
    %         +abs(get_distace(UE3_best4,UE3_true)))/3

    BS3_sum = [0,0];
    BS4_sum = [0,0];

    for i=1:1:size(BS3_best, 1)
        BS3_sum = BS3_sum + BS3_best(i, 2 * index_3_arr_final(i) - 1: 2 * index_3_arr_final(i));
        BS4_sum = BS4_sum + BS4_best(i, 2 * index_4_arr_final(i) - 1: 2 * index_4_arr_final(i));

    end
    BS3_final = BS3_sum / size(BS3_best, 1);
    BS4_final = BS4_sum / size(BS4_best, 1);

    error_BS = (abs(get_distace(BS3_final,BS3_true))+abs(get_distace(BS4_final,BS4_true)))/2;
    error_UE3 = (abs(get_distace(UE1_best3,UE1_true))...
            +abs(get_distace(UE2_best3,UE2_true))...
            +abs(get_distace(UE3_best3,UE3_true)))/3;
    %     error_UE4 = (abs(get_distace(UE1_best4,UE1_true))...
    %             +abs(get_distace(UE2_best4,UE2_true))...
    %             +abs(get_distace(UE3_best4,UE3_true)))/3
    toc
    data_bs_error(y) = error_BS;
    data_ue_error(y) = error_UE3;
    BS3_arr_final(y, 1:2) = BS3_final(1:2);
    BS4_arr_final(y, 1:2) = BS4_final(1:2);

% BS3_true
% BS4_true
% BS3_best
% BS4_best
end
BS_err = mean(data_bs_error)
UE_err = mean(data_ue_error)
BS3333 = mean(BS3_arr_final, 1)
BS4444 = mean(BS4_arr_final, 1)