function theta=TDOACHAN(A,p,sigma) 
% A is the coordinate of BSs 
%A��BSS������
% p is the range measurement 
%P�Ƿ�Χ����
% sigma is the the variance of TDOA measurement 
%sigma��TDOA�����ķ���
[m,~]=size(A); %size�õ�A����������ֵ��[m,~]��~��ʾռλ������ֻҪ��m��ֵ��
k=sum(A.^2,2); %����Aÿ��Ԫ�طֱ�ƽ�����õ��¾���������ͣ���Ϊ����K
G1=[A(2:end,:)-ones(m-1,1)*A(1,:),p]; %�õ�Xm1,Ym1,Rm1,��ֵ��mȡֵ[2,i],��������Ga
h1=1/2*(p.^2-k(2:end,:)+ones(m-1,1)*k(1,:)); %��������h
Q=diag(ones(m-1,1)*sigma); %����TDOA��Э�������
Q = cov(p);

% initial estimate 
theta0=inv(G1'*inv(Q)*G1)*G1'*inv(Q)*h1; %ͨ��һ��WLS�㷨������⣬
s=A(2:end,:)-ones(m-1,1)*theta0(1:2,:)'; 
d=sum(s.^2,2);%����sÿ��Ԫ�طֱ�ƽ�����õ��¾���������ͣ���Ϊ����d
B1=diag(d.^(1/2)); 
cov1=B1*Q*B1;
% first wls 
theta1=inv(G1'*inv(cov1)*G1)*G1'*inv(cov1)*h1; %���е�һ��WLS����
cov_theta1=inv(G1'*inv(cov1)*G1); %�õ�theta1��Э�������
% second wls 
G2=[1,0;0,1;1,1]; %����G'
h2=[(theta1(1,1)-A(1,1))^2;(theta1(2,1)-A(1,2))^2;theta1(3,1)^2]; %����h'
B2=diag([theta1(1,1)-A(1,1),theta1(2,1)-A(1,2),theta1(3,1)]); %����b'
cov2=4*B2*cov_theta1*B2; %�õ����ʸ����Э�������
theta2=inv(G2'*inv(cov2)*G2)*G2'*inv(cov2)*h2; %���������Ȼ���Ƶõ�
theta=theta2.^(1/2)+[A(1,1);A(1,2)]; %�õ�MSλ�õĹ���ֵ���꣬�Լ�����
theta=theta';%ת��Ϊ��x,y����ʽ

theta=abs(theta1(1:2)');%ת��Ϊ��x,y����ʽ

% a =1
