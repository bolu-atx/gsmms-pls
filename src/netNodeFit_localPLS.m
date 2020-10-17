% [gsom,y_hat,temp_err] = nodefit_coopPLS(gsom,training_data,b,bmus,Distance(b,:),Sigma(1));
% Takes training data, best matching units and the current GSOM and tries
% to fit a local PLS model
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
function [gsom, err] = nodefit_localPLS(gsom,training_data,current_node,bmus)
A = gsom.model_parameters.num_components;

% find data for the current node

% Since the clusters are local, we can't use the global mean and std
X = training_data.model_inputs(bmus == current_node,:);
Y = training_data.outputs(bmus == current_node,:);
[n, dx] = size(X);
if n < A+5;
    warning(sprintf('Not enough training data for node %d.',current_node));
    err = zeros(n,1);
else
    % Apply weighted PLS
    plsmodel = pls2(X,Y,A);
    

    res = plsmodel.y_residual; % residual to be only the output prediction residual
    err = sum(res.^2,2); % use the sum of squared PLS residuals as the error
    
    gsom.models{current_node} = plsmodel;
    plsmodel.activation = size(training_data.model_inputs,1);
    if size(err,2) == 1
        gsom.rmsReg(current_node) = sqrt(mean(err));
    else
        gsom.rmsReg(current_node) = sqrt(mean(sum(err,2)));
    end
    gsom.activation(current_node) = size(err,1);
end
end