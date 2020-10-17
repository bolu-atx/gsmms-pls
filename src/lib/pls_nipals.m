% THIS FUNCTION IS RETIRED
function [B,Wstar,T,P,Q,W,R2X,R2Y]=pls_nipals(X,Y,A,gamma)
%#
%# AIM:         performs PLS calibration on X and Y
%# PRINCIPLE:   Uses the NIPALS algorithm to perform PLS model calibration
%# REFERENCE:   Multivariate Calibration, H. Martens, T. Naes, Wiley and
%#              sons, 1989
%#
%# INPUT:
%# X            matrix of independent variables (e.g. spectra) (n x p)
%# Y            vector of y reference values (n x 1)
%# A            number of PLS factors to consider
%# gamma        X-weights penalty matrix (p x 1) - if 1, no penalty
%# 
%# OUTPUT:
%# B            regression coefficients (p x 1)
%# W            X-weights (p x A)
%# T            scores (n x A)
%# P            X-loadings (p x A)
%# Q            Y-loadings (A x 1)
%# R2X          percentage of X variance explained by each PLS factor
%# R2Y          percentage of Y-variance explained by each PLS factor
%#
%# AUTHOR:      Xavier Capron
%# 			    Copyright(c) 2004 for ChemoAC
%# 			    FABI, Vrije Universiteit Brussel
%# 			    Laarbeeklaan 103, 1090 Jette
%# 			    Belgium
%#             
%# VERSION: 1.0 (24/11/2004)
% moddification: preprocess parameter of the original code is removed by HDLi
% Modification: Bo Lu - Dow chemical comapny, freeport, tx. June 25th, 2013
if nargin<4;gamma = ones(size(X,2),1);end;

[n,p]=size(X);

Xorig=X;
Yorig=Y;

ssqX=sum(sum((X.^2)));
ssqY=sum(Y.^2);

for a=1:A
    W(:,a)=X'*Y; % w = X'*u
    W(:,a)=W(:,a)/norm(W(:,a)); % same as other
    T(:,a)=X*W(:,a); % same as other
    
    P(:,a)=X'*T(:,a)/(T(:,a)'*T(:,a)); %yup but this is not normalized
    Q(a,1)=Y'*T(:,a)/(T(:,a)'*T(:,a)); % this should be rows of 1s?
    
    X=X-T(:,a)*P(:,a)';
    Y=Y-T(:,a)*Q(a,1);
    R2X(a,1)=(T(:,a)'*T(:,a))*(P(:,a)'*P(:,a))/ssqX*100;
    R2Y(a,1)=(T(:,a)'*T(:,a))*(Q(a,1)'*Q(a,1))/ssqY*100;   
end

Wstar=W*inv(P'*W);
B=Wstar*Q;