% [gsom,y_hat,temp_err] = nodefit_coopPCA(gsom,training_data,b,bmus,Distance(b,:),Sigma(1));
% Takes training data, best matching units and the current GSOM and tries
% to fit a cooperative PCA model to it.
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
function [gsom, err] = nodefit_coopPCA(gsom,training_data,current_node,bmus,top_dis,coop)

% find the distance to the bmu of each training item
top_dis = top_dis(bmus);

% compute the weights of each training item by accounting for
% topological distance from neighboring nodes
W = exp(-top_dis.^2/(coop^2));

% apply weighted PCA
%[T,P,eigv,varpc,res] = pca_svd(training_data.inputs, 2, W);
[P,T,eigs] = pca(training_data.model_inputs,'NumComponents',2,'Weights',W);
res = training_data.model_inputs - T*P';

% this is the code for multi-linear regression
%     for i = 1:gsom.trainData.no
%         inds = [find(mask(i,:)==1),size(X,2)];  % the mask tells us what regressors affect the ith output!
%         Xtemp = X(:,inds);
%         WX = repmat(W,[1,size(Xtemp,2)]).*Xtemp;
%         WY = W.*Y(:,i);
%         gsom.coefs{current_node}(i,inds) = (Xtemp'*WX)\(Xtemp'*WY);
%         y_hat(:,i) = X*gsom.coefs{current_node}(i,:)';
%     end

% compute regional errors (and global error)
err = sum(res.^2,2); % use the sum of squared PCA residuals as the error
err = err(bmus==current_node,:); % only report errors for which current_node 
                                 % is the best matching unit
pcamodel.W = W;
pcamodel.T = T;
pcamodel.eigs = eigs;
pcamodel.P = P;
gsom.models{current_node} = pcamodel;
pcamodel.activation = size(training_data.model_inputs,1);

if size(err,2) == 1
    gsom.rmsReg(current_node) = mean(sqrt((err)));
else
    gsom.rmsReg(current_node) = mean(sqrt(sum(err,2)));
end
gsom.activation(current_node) = size(err,1);

end