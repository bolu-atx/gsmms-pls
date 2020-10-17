function PLS=pls2(X,y,A,weights,var_sel,method,name)
%+++  PLS=pls2(x0,y0,A,weights,var_sel,method,name);
% Sample weighted PLS regression algorithm
%
%+++  Input:
%     X,y: sample data and y-value to predict
%     A: number of PLS components
%     weights: vector dimension (N,1) containing weight of each
%           observation, default is 1 for each element.
%     method: pretreat method for X, either "center" or "autoscaling". y is
%             always centered in our libPLS package.
%+++  Ouput : is a structural array which are explained at the end of this code
%
% Modified by Bo Lu, June 4th, 2013
% Last revision: Aug 23, 2014

if nargin<7;name='PLS model';end
if nargin<6;method='autoscaling';end
if nargin<5;var_sel = 1:size(X,2);end
if nargin<4;weights = ones(size(X,1),1);end;
if nargin<3;A=2;end;
if size(var_sel,1)>size(var_sel,2);var_sel = var_sel';end;
[n,p]=size(X);
A=min([n p A]); % ensures the # of components is valid

if (size(weights,2) > size(weights,1))
    weights = weights';
end

%+++ data pretreatment, para1 is mean, para2 is std
[Xs,xpara1,xpara2]=pretreat_w(X(:,var_sel),method,weights);
[ys,ypara1,ypara2]=pretreat_w(y,method,weights);
p_reduced = size(Xs,2);

% THIS FUNCTION HAS BEEN RETIRED IN FAVOR OF A BETTER IMPLEMENTATION
% [B,Wstar,T,P,Q,W,R2X,R2Y]=pls_nipals(Xs,ys,A);
%[B,P,Q,W,T,U,inner,Xres,yres]=mvplsnipals(Xs,ys,A);
[P,Q,T,U,R,B,pctVar,mse,stats] = weightedPLS(Xs,ys,A,'sample_weights',weights);
% convert the scores and loadings to the standard NIPALS format
B = B(2:end);
W = zeros(size(R));
for i = 1:size(P,2);
    %x-scores and loadings
    pnorm = norm(P(:,i));
    T(:,i) = T(:,i)*pnorm;
    P(:,i) = P(:,i)/pnorm;
    W(:,i) = R(:,i)*pnorm;    
    % y-scores and loadings
    qnorm = norm(Q(:,i));
    Q(:,i) = Q(:,i)/qnorm;
    U(:,i) = U(:,i)*qnorm;
end

% T is normalized in weightedPLS
% P is un-normalized in weightedPLS
% TP_simpls = TP_nipals
% Q is un-normalized in weightedPLS
% U is normalized in weightedPLS
% B is equivalent to B in mvplsnipals
%
% R is similar to Wstar but not equivalent
% Beta = R*Q'
% U = U2 ./ Q2
% Q in the NIPALS is always 1, it's been Q-normalized

VIP=vip(Xs,ys,T,W);

%+++ get regression coefficients that link X and y (original data) ************
coef=zeros(p+1,1); % +1 is for the bias
C=zeros(numel(var_sel),1); 

% Obtain regression coefficients for dataset without autoscaling
C = ypara2*B./xpara2';
coef([var_sel p+1],1) = [C;ypara1-xpara1*C;]; % intercept


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
y_pred=[X ones(n,1)]*coef(:,1); % predict use the maximum number of components
y_residual=y_pred-y;
%********************************************
% Need to compensate for the weights in evaluating the R2.
SST=sum((y-ypara1).^2.*weights);
SSR=sum((y_pred-ypara1).^2.*weights);
SSE=sum((y-y_pred).^2.*weights);
R2=1-SSE/SST;

% Original correlation doesn't work, need a weighted version... too lazy,
% TODO
% corr_matrix = corrcoef(y, y_pred);
% R2corr = corr_matrix(1,2);


%+++ Output**************************************
PLS.method=method;
PLS.name = name;
% PLS.Z_x=Xs;
% PLS.Z_y=ys;
PLS.algorithm = 'SIMPLS';
PLS.beta = zeros(p,1);
PLS.beta(var_sel)=B; %applies to autoscaled inputs
PLS.B =coef(:,1); % applies to raw inputs
PLS.xpara1 = xpara1;
PLS.xpara2 = xpara2;
PLS.ypara1 = ypara1;
PLS.ypara2 = ypara2;
PLS.T=T;
PLS.P=P;
PLS.Q=Q;
PLS.VIP=VIP;
PLS.W=W;
PLS.sample_weights = weights;
PLS.R = R;
PLS.Wstar=R; % approximation
PLS.y_predicted=y_pred;
PLS.y_residual=y_residual;
PLS.SST=SST;
PLS.SSR=SSR;
PLS.SSE=SSE;
PLS.RMSEF=sqrt(SSE/n);
PLS.R2=R2; % unadjusted classical R2 (accounts for bias)
%PLS.R2corr = R2corr; % adjusted correlation based R2 (ignores bias)
PLS.bias = mean(y-y_pred);
PLS.var_sel = var_sel; % variable used in the regression.
end