% [gsom,y_hat,temp_err] = nodefit_PCA(gsom,training_data,b,bmus,Distance(b,:),Sigma(1));
% Takes training data, best matching units and the current GSOM and tries
% to fit a local PCA model to it.
%
% ++INPUT++
%   gsom - a data structure containing all the associated SOM parameters
%   training_data - a data structure containing the input, output variables
%   current_node - a integer denoting the current node being trained for
%   bmus - indices of the best matching units, for each training sample
%   top_dis - topological distances of the adjacent neighbours to the
%           current node
%   coop - cooperation parameter (1 being most cooperative, 0 being no
%   cooperation)
%
function [gsom, err] = nodefit_localPCA(gsom,training_data,current_node,bmus)

X = training_data.model_inputs(bmus == current_node,:);
X = X-repmat(mean(X),size(X,1),1);
% apply weighted PCA
[P,T,eigv] = pca(X, 'NumComponents',2);
res = X - T*P';

%     for i = 1:gsom.trainData.no
%         inds = [find(mask(i,:)==1),size(X,2)];  % the mask tells us what regressors affect the ith output!
%         Xtemp = X(:,inds);
%         WX = repmat(W,[1,size(Xtemp,2)]).*Xtemp;
%         WY = W.*Y(:,i);
%         gsom.coefs{current_node}(i,inds) = (Xtemp'*WX)\(Xtemp'*WY);
%         y_hat(:,i) = X*gsom.coefs{current_node}(i,:)';
%     end

% compute regional errors (and global error)
err = sumsqr(res')'; % use the sum of squared PCA residuals as the error
pcamodel.T = T;
pcamodel.eigs = eigv;
pcamodel.P = P;
pcamodel.activation = size(X,1);
gsom.models{current_node} = pcamodel;

gsom.rmsReg(current_node) = sqrt(mean(err));
gsom.activation(current_node) = size(err,1);
end