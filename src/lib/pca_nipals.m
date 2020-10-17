function [T,P]= pca_nipals(X,A)
% NIPALS
% Performs NIPALS iterative calculation of the PCA components
% Author: Bo Lu June 30th, 2014
[n,p]=size(X);
tol = 1e-5;

for a=1:A
    T_prev = zeros(n,1);
    T(:,a)=randn(n,1);
    while norm(T(:,a) - T_prev) > tol;
        T_prev = T(:,a);
        P(:,a) = X'*T(:,a)/(T(:,a)'*T(:,a));
        P(:,a)=P(:,a)/norm(P(:,a));
        T(:,a) = X*P(:,a);
    end
    X=X-T(:,a)*P(:,a)';
end