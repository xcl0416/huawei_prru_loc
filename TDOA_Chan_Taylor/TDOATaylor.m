function theta=TDOATaylor(A,p,sigma) 
% A is the coordinate of BSs
 % p is the range measurement 
% sigma is the the variance of TOA measurement 
% initial estimate 
theta0=TDOACHAN(A,p,sigma); %����TDOACHAN�õ�һ����ʼ�Ĺ���λ��
% theta0=TDOACHAN1(A,p,sigma); %����TDOACHAN�õ�һ����ʼ�Ĺ���λ��

theta0 = real(theta0);
thetachan = theta0;
delta=norm(theta0); %�õ�����
n = 0;
while norm(delta)>1e-2 %�õ��㹻С��ֵ
       [m,~]=size(A); %size�õ�A����������ֵ��[m,~]��~��ʾռλ������ֻҪ��m��ֵ��
       d=sum((A-ones(m,1)*theta0).^2,2); 
       R=d.^(1/2); 
       G1=ones(m-1,1)*(A(1,1)-theta0(1,1))/R(1,1)-(A(2:m,1)-theta0(1,1))./R(2:m,:);        
       G2=ones(m-1,1)*(A(1,2)-theta0(1,2))/R(1,1)-(A(2:m,2)-theta0(1,2))./R(2:m,:);        
       G=[G1,G2]; %����Gt
       h=p-(R(2:m,:)-ones(m-1,1)*R(1,:)); %����Ht
       Q=diag(ones(m-1,1)*sigma); %TDOA����ֵ��Э�������
       Q = cov(p);

       delta=((G'/Q)*G)\((G'/Q)*h); %��Ȩ��С���˽�
%        delta=pinv(G'*pinv(Q)*G)*G'*pinv(Q)*h; %��Ȩ��С���˽�
%        delta=(G'*lsqminnorm(Q, eye)*G)\(G'*lsqminnorm(Q, eye)*h); %��Ȩ��С���˽�


       theta0=theta0+delta'; %�ۼ�
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
