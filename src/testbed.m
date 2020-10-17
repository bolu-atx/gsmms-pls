data_import_and_pretreat
Xs = pretreat(X,'autoscaling');
Ys = pretreat(Y,'autoscaling');


m1 = 1:3000;
m2 = 3001:5034;
A = 6;
%Xs = scores(:,1:10);
PLSmodel1 = pls(Xs(m1,:),Ys(m1),A);
PLSmodel2 = pls(Xs(m2,:),Ys(m2),A);
PLSmodel3 = pls_merge(PLSmodel1,PLSmodel2,1,1);
t0_1 = PLSmodel1.T;
p0_1 = PLSmodel1.P;
inner0_1 = PLSmodel1.inner;
t0_2 = PLSmodel2.T;
p0_2 = PLSmodel2.P;
inner0_2 = PLSmodel2.inner;
L1 = diag(sqrt(diag(t0_1'*t0_1)));
R1 = p0_1*L1;
S1 = inner0_1*L1;
L2 = diag(sqrt(diag(t0_2'*t0_2)));
R2 = p0_2*L2;
S2 = inner0_2*L2;
% Create new calibration set
lambda1 = 1
lambda2 = 1

Xnew = [lambda1*R1';lambda2*R2'];
ynew = [lambda1*S1';lambda2*S2'];
warning off;
[B,P,Q,W,T,U,inner,Xres,yres]=mvplsnipals(Xnew,ynew,A);
warning on;
Wstar = W*inv(P'*W);
PLSmodeltotal = pls(Xs,Ys,A);

B
PLSmodeltotal.B_full