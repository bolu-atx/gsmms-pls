% PCA_svd
% Apply SVD to performing PCA
function [T,P,eigv,varpc,res] = pca_svd(X, maxfact, W)
[m,n] = size(X);
if nargin < 3;W = ones(size(X,1),1);end;
if numel(W) ~= size(X,1)
    error('The number of rows for the weighing matrix does not equal to the input.');
end

W = diag(W);

if (nargin == 2) | (nargin == 3),
  % cannot find more PC's than number of variables
  k = min([n, maxfact]);
elseif (nargin == 1) | (isempty(k) == 1),
  k = n;
end

if abs(max(mean(X))) > 1e3*eps,
  warning('Input matrix is not mean centered! Please consider doing this!');
end

% 1. SVD
XX = X'*W'*W*X;
[U,S,VT] = svd(XX);

% Extract the eigenvalues.
eigv=diag(S);
nocomp = length(eigv);

if k < nocomp,
  eigv=eigv(1:k);
else
  k = nocomp;
end

% Find the loading vectors.
P = VT(:,1:k);
% Calculate the scores.
T = X*P;

% VARPC is the X variance captured in each factor.
varexp = (eigv(1:k)/sum(eigv));
for a=1:k, % The accumulated variance explained for each PC
  varpc(a)=sum(varexp(1:a));
end

% The residual matrix.
res = X - T*P';
end