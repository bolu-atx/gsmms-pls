function PLSmodel=pls_recursive(PLSmodel,X,y,lambda)
%+++  Input:
%    PLSmodel - An existing PLS model to be updated.
%    X,y      - incoming sample data and y-value to predict
%    lambda   - is the forgetting factor
%+++  Ouput
%    PLS      - a PLS structural array
%
% Author:
% Bo Lu, The University of Texas-Austin
% bluevires@gmail.com
% 
% Reference: Helland's paper on recursive PLS
%
% Create representation of data based on loadings and new X and y
if nargin < 4;lambda = 1;end
Xs = pretreat(X,'autoscaling',PLSmodel.xpara1,PLSmodel.xpara2);
ys = pretreat(y,'autoscaling',PLSmodel.ypara1,PLSmodel.ypara2);

A = size(PLSmodel.T,2);
n = size(X,1);
p = size(X,2);
t0 = PLSmodel.T;
p0 = PLSmodel.P;
inner0 = PLSmodel.inner;

L = diag(sqrt(diag(t0'*t0)));
R = p0*L;
S = inner0*L;

% Create new calibration set
Xnew = [lambda*R';Xs];
ynew = [lambda*S';ys];
warning off;
[B,P,Q,W,T,U,inner,Xres,yres]=mvplsnipals(Xnew,ynew,A);
warning on;
Wstar = W*inv(P'*W);


% notice that here, B is the regression coefficients linking the scaled
% X and y
% VIP=vip(Xs,ys,T,W);

%+++ get regression coefficients that link X and y (original data) ************
coef=zeros(p+1,A);
C=zeros(numel(PLSmodel.var_sel),1);

% Obtain regression coefficients for dataset without autoscaling
for j = 1:A
    %Bj = sum(Wstar(:,1:j)*(diag(inner(1:j))*diag(Q(1:j))),2);
    Bj = B(j,:)';
    C = PLSmodel.ypara2*Bj./PLSmodel.xpara2';
    coef([PLSmodel.var_sel p+1],j) = [C;PLSmodel.ypara1-PLSmodel.xpara1*C;]; % intercept
end


% predict only the current sample
y_pred=[X ones(n,1)]*coef(:,end); 
y_residual=y_pred-y;

% combine current prediction with past prediction results
PLSmodel.y_predicted = [PLSmodel.y_predicted;y_pred];
PLSmodel.y_residual = [PLSmodel.y_residual;y_residual];

SST=PLSmodel.SST+(y-mean(PLSmodel.y_predicted)).^2;
SSR=PLSmodel.SSR+sum(y_pred-mean(PLSmodel.y_predicted)).^2;
SSE=PLSmodel.SSE+y_residual.^2;

R2=1-SSE/SST;
% this requires us saving the past Ys
% corr_matrix = corrcoef(y, y_pred);
% R2corr = corr_matrix(1,2);

%+++ Output**************************************
PLSmodel.beta = zeros(p,1);
PLSmodel.beta(PLSmodel.var_sel)=B(end,:); %applies to autoscaled inputs
PLSmodel.B_full=coef;
PLSmodel.B =coef(:,end); % applies to raw inputs
PLSmodel.T=T;
PLSmodel.P=P;
PLSmodel.Q=Q;
PLSmodel.W=W;
PLSmodel.inner = inner;
%PLS.Wstar=Wstar;
PLSmodel.SST=SST;
PLSmodel.SSR=SSR;
PLSmodel.SSE=SSE;
PLSmodel.R2=R2; % unadjusted classical R2 (accounts for bias)
% PLS.R2corr = R2corr; % adjusted correlation based R2 (ignores bias)
% PLS.bias = mean(y-y_pred);
end
