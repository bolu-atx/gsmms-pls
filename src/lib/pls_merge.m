% PLS_merge
% Merge two PLS models together
function PLSmodel=pls_merge(PLSmodel1,PLSmodel2,lambda1,lambda2)
%+++  Input:
%    PLSmodel, PLSmodel2 - structural PLS arrays
%    lambda - weighing for the respective models
%+++  Ouput
%    PLS      - a PLS structural array
%
% Author:
% Bo Lu, The University of Texas-Austin
% bluevires@gmail.com
% 
% Reference: Helland's paper on recursive PLS
% Create representation of data based on loadings and new X and y
%
%
% WHAT IF THE TWO MODELS DO NOT HAVE THE SAME MEAN and VARIANCE?
%
A1 = size(PLSmodel1.T,2);
A2 = size(PLSmodel2.T,2);
if A1 ~= A2
    error('The number of components are different for the two models.')
else
    A = A1;
end
p = size(PLSmodel1.beta,1);

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
Xnew = [lambda1*R1';lambda2*R2'];
ynew = [lambda1*S1';lambda2*S2'];

warning off;
[B,P,Q,W,T,U,inner,Xres,yres]=mvplsnipals(Xnew,ynew,A);
warning on;
Wstar = W*inv(P'*W);


% notice that here, B is the regression coefficients linking the scaled
% X and y
% VIP=vip(Xs,ys,T,W);

%+++ get regression coefficients that link X and y (original data) ************
coef=zeros(p+1,A);
C=zeros(numel(PLSmodel1.var_sel),1);

% Obtain regression coefficients for dataset without autoscaling
for j = 1:A
    %Bj = sum(Wstar(:,1:j)*(diag(inner(1:j))*diag(Q(1:j))),2);
    Bj = B(j,:)';
    C = PLSmodel1.ypara2*Bj./PLSmodel1.xpara2';
    coef([PLSmodel1.var_sel p+1],j) = [C;PLSmodel1.ypara1-PLSmodel1.xpara1*C;]; % intercept
end

% this requires us saving the past Ys
% corr_matrix = corrcoef(y, y_pred);
% R2corr = corr_matrix(1,2);
%+++ Output**************************************
PLSmodel.beta = zeros(p,1);
PLSmodel.beta(PLSmodel1.var_sel)=B(end,:); %applies to autoscaled inputs
PLSmodel.B_full=B;
PLSmodel.B =coef(:,end); % applies to raw inputs
PLSmodel.T=T;
PLSmodel.P=P;
PLSmodel.Q=Q;
PLSmodel.W=W;
PLSmodel.inner = inner;
%PLS.Wstar=Wstar;

end