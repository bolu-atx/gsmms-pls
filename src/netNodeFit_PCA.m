% [gsom,y_hat,temp_err] = nodefit_PCA(gsom,training_data,b,bmus,Distance(b,:),Sigma(1));
% Takes training data, best matching units and the current GSOM and tries
% to fit a local PCA model to it.
function [gsom, y_hat, err] = nodefit_PCA(gsom,training_data,current_node,bmus,top_dis,coop)
                
    % find the distance to the bmu of each training item
    top_dis = top_dis(bmus);
   
    % compute the weights of each training item
    W = exp(-top_dis.^2/(2*coop^2));
    W = reshape(W,numel(W),1);

    % compute the least squares solution
    mask = gsom.trainData.mask;
    y_hat = zeros(size(X,1),size(Y,2));
    gsom.coefs{current_node} = zeros(size(mask));
    for i = 1:gsom.trainData.no
        inds = [find(mask(i,:)==1),size(X,2)];  % the mask tells us what regressors affect the ith output!
        Xtemp = X(:,inds);
        WX = repmat(W,[1,size(Xtemp,2)]).*Xtemp;
        WY = W.*Y(:,i);
        gsom.coefs{current_node}(i,inds) = (Xtemp'*WX)\(Xtemp'*WY);
        y_hat(:,i) = X*gsom.coefs{current_node}(i,:)';
    end

    % compute regional errors (and global error)
    err = Y-y_hat;
    err = err(bmus==current_node,:); % only report errors for which current_node is the best matching unit
    gsom.rmsReg(current_node) = sqrt(mean( sum(err.^2,2) ));
    gsom.activation(current_node) = size(err,1);
end