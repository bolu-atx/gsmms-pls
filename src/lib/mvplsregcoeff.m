function [b] = mvplsregcoeff(p,q,w,inner)
%MVPLSREGCOEFF -- Calculate PLSR regression coefficients
%	
%  Usage:
%    [b] = mvplsregcoeff(p,q,w,inner)
%	
%  Inputs:
%    p        X loadings
%    q        y loadings (vector for PLS1, matrix for PLS2)
%    w        loading weights
%    inner    vector containing the inner (X/Y scores) relationships
%
%  Outputs:
%    b        regression coefficients
%
%  Description:
%    Calculates Partial Least Squares regression coefficients from
%    X/y loadings, loading weights and X/y inner relations.
%

% Is it PLS1 or PLS2?
[rq,cq] = size(q);
if rq == 1,
  b  = w*inv(p'*w)*diag(inner)*diag(q);
  % make output like pls_toolbox
  b  = cumsum(b',1);
else
  b  = w*inv(p'*w)*diag(inner)*q';  
end
