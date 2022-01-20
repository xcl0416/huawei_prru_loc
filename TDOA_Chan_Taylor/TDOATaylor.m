function theta=TDOATaylor(A,p,sigma) 
% A is the coordinate of BSs
 % p is the range measurement 
% sigma is the the variance of TOA measurement 
% initial estimate 
theta0=TDOACHAN(A,p,sigma); %调用TDOACHAN得到一个初始的估计位置
% theta0=TDOACHAN1(A,p,sigma); %调用TDOACHAN得到一个初始的估计位置

theta0 = real(theta0);
thetachan = theta0;
delta=norm(theta0); %得到范数
n = 0;
while norm(delta)>1e-2 %得到足够小的值
       [m,~]=size(A); %size得到A的行列数赋值给[m,~]，~表示占位，就是只要行m的值！
       d=sum((A-ones(m,1)*theta0).^2,2); 
       R=d.^(1/2); 
       G1=ones(m-1,1)*(A(1,1)-theta0(1,1))/R(1,1)-(A(2:m,1)-theta0(1,1))./R(2:m,:);        
       G2=ones(m-1,1)*(A(1,2)-theta0(1,2))/R(1,1)-(A(2:m,2)-theta0(1,2))./R(2:m,:);        
       G=[G1,G2]; %构建Gt
       h=p-(R(2:m,:)-ones(m-1,1)*R(1,:)); %构建Ht
       Q=diag(ones(m-1,1)*sigma); %TDOA测量值的协方差矩阵
       Q = cov(p);

       delta=((G'/Q)*G)\((G'/Q)*h); %加权最小二乘解
%        delta=pinv(G'*pinv(Q)*G)*G'*pinv(Q)*h; %加权最小二乘解
%        delta=(G'*lsqminnorm(Q, eye)*G)\(G'*lsqminnorm(Q, eye)*h); %加权最小二乘解


       theta0=theta0+delta'; %累加
       n = n + 1;
       if n > 10
           break
       end
end 
if isnan(theta0)
    theta = thetachan;
else
    theta = theta0;
if theta0(1) > 1000 || theta0(1) < -1000 || theta0(2) > 1000 || theta0(2) < -1000
    theta = thetachan;
end

end
