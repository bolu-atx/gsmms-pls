function [y_pred,RMSEP,diagnostics]=plsval(plsmodel,Xtest,ytest,nLV)
%+++ Compute prediction errors on a test set
%+++ plsmodel: a structural data obtained from the function pls.m in this directory;
%+++ Xtest: test samples
%+++ ytest: y values for test samples (for "really" new samples, no y values available
%+++ nLV: number of latent variables for calibration models.
%+++ Hongdong Li,Oct.21,2007;

%*** Modified by Bo Lu
%*** Added T^2 and Q calculation

if nargin<4;
    nLV=size(plsmodel.T,2);
end;

Xs = pretreat(Xtest(:,plsmodel.var_sel),'autoscaling',plsmodel.xpara1,plsmodel.xpara2);
Ys = pretreat(ytest,'autoscaling',plsmodel.ypara1,plsmodel.ypara2);

Xtest=[Xtest ones(size(Xtest,1),1)];
%ypred=Xtest*plsmodel.regcoef_original_all(:,nLV);
y_pred=Xtest*plsmodel.B;

SST=sum((ytest-mean(ytest)).^2);
SSR=sum((y_pred-mean(ytest)).^2);
SSE=sum((ytest-y_pred).^2);
diagnostics.R2=1-SSE/SST;

% SPE
Xtest_pred = Xs*plsmodel.W*plsmodel.P';
Xresidual = Xs - Xtest_pred;
diagnostics.SPE = sqrt(sum(Xresidual.^2,2));

% T2
Ttest = Xs*plsmodel.Wstar;
diagnostics.T2 = sum( bsxfun(@rdivide, abs(Ttest).^2, var(Ttest,[],1)) , 2);
RMSEP=sqrt(sum((y_pred-ytest).^2)/length(ytest));