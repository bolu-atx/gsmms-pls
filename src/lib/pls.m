function PLS=pls(X,y,A,var_sel,method,name)
%+++  PLS=pls(x0,y0,A,method);
%+++  programmed based on the 'pls_nipals.m' in the package of new chemo_AC
%+++  developted by Massart's Group.

%+++  Input:
%     X,y: sample data and y-value to predict
%     A: number of PLS components
%     method: pretreat method for X, either "center" or "autoscaling". y is
%             always centered in our libPLS package.
%     gamma - collinearity detection penalty matrix (see
%             check_redundant_variables.m)

%+++  Ouput : is a structural array which are explained at the end of this code

%+++  Hongdong Li, June 1,2008.
%+++  Contact: lhdcsu@gmail.com.
% Modified by Bo Lu, June 4th, 2013
% The Dow Chemical Company

% Last revision: Aug 23, 2013

if nargin<6;name='PLS model';end
if nargin<5;method='autoscaling';end
if nargin<4;var_sel = 1:size(X,2);end
if nargin<3;A=2;end;
if size(var_sel,1)>size(var_sel,2);var_sel = var_sel';end;
[n,p]=size(X);
A=min([n p A]); % ensures the # of components is valid

%+++ data pretreatment, para1 is mean, para2 is std
[Xs,xpara1,xpara2]=pretreat(X(:,var_sel),method);
[ys,ypara1,ypara2]=pretreat(y,method);
p_reduced = size(Xs,2);

% THIS FUNCTION HAS BEEN RETIRED IN FAVOR OF A BETTER IMPLEMENTATION
% [B,Wstar,T,P,Q,W,R2X,R2Y]=pls_nipals(Xs,ys,A);
[B,P,Q,W,T,U,inner,Xres,yres]=mvplsnipals(Xs,ys,A);

Wstar = W*inv(P'*W);


% notice that here, B is the regression coefficients linking the scaled
% X and y
VIP=vip(Xs,ys,T,W);

%+++ get regression coefficients that link X and y (original data) ************
coef=zeros(p+1,A);
C=zeros(numel(var_sel),1);

% Obtain regression coefficients for dataset without autoscaling
for j = 1:A
    %Bj = sum(Wstar(:,1:j)*(diag(inner(1:j))*diag(Q(1:j))),2);
    Bj = B(j,:)';
    C = ypara2*Bj./xpara2';
    coef([var_sel p+1],j) = [C;ypara1-xpara1*C;]; % intercept
end


% % +++ Calculate training T^2, Q diagnostics ********************************
% T2 = sum( bsxfun(@rdivide, abs(T).^2, var(T,[],1)) , 2);
% Xresidual = Xs - T*P';
% SPE = sqrt(sum(Xresidual.^2,2));
% 
% % T control limit
% tcrit = (2*(n^2-1)/(n*(n-2)))*...
%     [finv(0.999,2,n-2),...
%     finv(0.99,2,n-2),...
%     finv(0.95,2,n-2)];
% 
% % SPE control limit
% SPEcrit = sqrt(sum(sum(Xresidual.^2))./((n-3)*(p_reduced-2)))*...
%     sqrt([finv(0.999,p_reduced-2,((n-3)*(p_reduced-2))),...
%     finv(0.99,p_reduced-2,((n-3)*(p_reduced-2))),...
%     finv(0.95,p_reduced-2,((n-3)*(p_reduced-2)))]);
% PLS.T2 = T2;
% PLS.T2crit = tcrit; % control limits: 3 values corresponding to 95% 99% and 99.9% limits
% PLS.SPEcrit = SPEcrit; % control limits: 3 values corresponding to 95% 99% and 99.9% limits
% PLS.SPE = SPE;
% 
% % +++ Calculate Selectivity Ratio ****************************************
% PLS.Sratio= zeros(p,1);
% Sratio = var(T*P')./var(Xresidual);
% PLS.Sratio(var_sel) = Sratio; % Selectivity ratio

%+++ ********************************************
y_pred=[X ones(n,1)]*coef(:,end); % predict use the maximum number of components
y_residual=y_pred-y;
%********************************************
SST=sum((y-mean(y)).^2);
SSR=sum((y_pred-mean(y)).^2);
SSE=sum((y-y_pred).^2);
R2=1-SSE/SST;
corr_matrix = corrcoef(y, y_pred);
R2corr = corr_matrix(1,2);

%+++ Output**************************************
PLS.method=method;
PLS.name = name;
% PLS.Z_x=Xs;
% PLS.Z_y=ys;

PLS.beta = zeros(p,1);
PLS.beta(var_sel)=B(end,:); %applies to autoscaled inputs
PLS.B_full=B;
PLS.B =coef(:,end); % applies to raw inputs
PLS.xpara1 = xpara1;
PLS.xpara2 = xpara2;
PLS.ypara1 = ypara1;
PLS.ypara2 = ypara2;
PLS.T=T;
PLS.P=P;
PLS.Q=Q;
PLS.VIP=VIP;
PLS.W=W;
PLS.inner = inner;

PLS.Wstar=Wstar;
PLS.y_predicted=y_pred;
PLS.y_residual=y_residual;
PLS.SST=SST;
PLS.SSR=SSR;
PLS.SSE=SSE;
PLS.RMSEF=sqrt(SSE/n);
PLS.R2=R2; % unadjusted classical R2 (accounts for bias)
PLS.R2corr = R2corr; % adjusted correlation based R2 (ignores bias)
PLS.bias = mean(y-y_pred);
PLS.var_sel = var_sel; % variable used in the regression.
end