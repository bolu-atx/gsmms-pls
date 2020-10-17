% Apply weighted mean
% X - matrix or a vector
% weights - self explanatory
function mean = wmean(X,weights)
[n dx] =size(X);
mean = sum(X.*repmat(weights,1,dx),1)./repmat(sum(weights),1,dx);
end