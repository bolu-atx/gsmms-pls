% ++ weighted least squares
%   weighted by topological distance and the cooperation rate
%    - higher cooperation rate - larger change
%    - higher distance - lower weight

function [MMS,y_hat,err] = pwa_wls_mimo(MMS,c,bmus,X,Y,top_dis,coop)
                
    % find the distance to the bmu of each training item
    top_dis = top_dis(bmus);
   
    % compute the weights of each training item
    W = exp(-top_dis.^2/(2*coop^2));
    W = reshape(W,numel(W),1);

    % compute the least squares solution
    mask = MMS.trainData.mask;
    y_hat = zeros(size(X,1),size(Y,2));
    MMS.coefs{c} = zeros(size(mask));
    for i = 1:MMS.trainData.no
        inds = [find(mask(i,:)==1),size(X,2)];  % the mask tells us what regressors affect the ith output!
        Xtemp = X(:,inds);
        WX = repmat(W,[1,size(Xtemp,2)]).*Xtemp;
        WY = W.*Y(:,i);
        MMS.coefs{c}(i,inds) = (Xtemp'*WX)\(Xtemp'*WY);
        y_hat(:,i) = X*MMS.coefs{c}(i,:)';
    end

    % compute regional errors (and global error)
    err = Y-y_hat;
    err = err(bmus==c,:); % only report errors for which c is the best matching unit
    MMS.rmsReg(c) = sqrt(mean( sum(err.^2,2) ));
    MMS.activation(c) = size(err,1);
end