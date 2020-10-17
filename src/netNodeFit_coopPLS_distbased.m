% [gsom,y_hat,temp_err] = nodefit_coopPLS(gsom,training_data,b,bmus,Distance(b,:),Sigma(1));
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
function [gsom, err] = nodefit_coopPLS(gsom,training_data,current_node,bmus,top_dis,coop)
A = gsom.model_parameters.num_components;
dis_local = dis(bmus == current_node);

sigma = prctile(dis_local,75)*4;
wj = exp(-dis.^2/(2*(sigma*(coop+0.01))^2)); % this is a modified Gaussian

% Apply weighted PLS
plsmodel = pls2(training_data.model_inputs,training_data.outputs,A,wj);

% does this work? no idea
% OLD PCA code
% =================
% [P,T,eigs] = pca(training_data.inputs,'NumComponents',2,'Weights',W);
% res = training_data.inputs - T*P';
% compute regional errors (and global error)
res = plsmodel.y_residual;
err = sum(res.^2,2); % use the sum of squared PLS residuals as the error

err = err(bmus==current_node,:); % only report errors for which current_node 
                                 % is the best matching unit
                                 % otherwise in cooperative training, all
                                 % the errors will be reported.
                                 
gsom.models{current_node} = plsmodel;
plsmodel.activation = size(training_data.model_inputs,1);

if size(err,2) == 1
    gsom.rmsReg(current_node) = sqrt(mean(err));
else
    gsom.rmsReg(current_node) = sqrt(mean(sum(err,2)));
end

gsom.activation(current_node) = size(err,1);
end