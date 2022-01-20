function theta=TDOACHAN(A,p,sigma) 
% A is the coordinate of BSs 
%A是BSS的坐标
% p is the range measurement 
%P是范围测量
% sigma is the the variance of TDOA measurement 
%sigma是TDOA测量的方差
[m,~]=size(A); %size得到A的行列数赋值给[m,~]，~表示占位，就是只要行m的值！
k=sum(A.^2,2); %矩阵A每个元素分别平方，得到新矩阵，在行求和，最为矩阵K
G1=[A(2:end,:)-ones(m-1,1)*A(1,:),p]; %得到Xm1,Ym1,Rm1,的值，m取值[2,i],构建矩阵Ga
h1=1/2*(p.^2-k(2:end,:)+ones(m-1,1)*k(1,:)); %构建矩阵h
Q=diag(ones(m-1,1)*sigma); %构建TDOA的协方差矩阵
Q = cov(p);

% initial estimate 
theta0=inv(G1'*inv(Q)*G1)*G1'*inv(Q)*h1; %通过一次WLS算法进行求解，
s=A(2:end,:)-ones(m-1,1)*theta0(1:2,:)'; 
d=sum(s.^2,2);%矩阵s每个元素分别平方，得到新矩阵，在行求和，最为矩阵d
B1=diag(d.^(1/2)); 
cov1=B1*Q*B1;
% first wls 
theta1=inv(G1'*inv(cov1)*G1)*G1'*inv(cov1)*h1; %进行第一次WLS计算
cov_theta1=inv(G1'*inv(cov1)*G1); %得到theta1的协方差矩阵
% second wls 
G2=[1,0;0,1;1,1]; %构建G'
h2=[(theta1(1,1)-A(1,1))^2;(theta1(2,1)-A(1,2))^2;theta1(3,1)^2]; %构建h'
B2=diag([theta1(1,1)-A(1,1),theta1(2,1)-A(1,2),theta1(3,1)]); %构建b'
cov2=4*B2*cov_theta1*B2; %得到误差矢量的协方差矩阵。
theta2=inv(G2'*inv(cov2)*G2)*G2'*inv(cov2)*h2; %运用最大似然估计得到
theta=theta2.^(1/2)+[A(1,1);A(1,2)]; %得到MS位置的估计值坐标，以及符号
theta=theta';%转换为（x,y）形式

theta=abs(theta1(1:2)');%转换为（x,y）形式

% a =1
