% function [gsom, err] = netNodeRecursive_localPLS(gsom,update_data,current_node,bmus)
% takes update data and recursively updates the local model corresponding
% to the current_node
%
% ++INPUT++
%   gsom - a data structure containing all the associated SOM parameters
%   update_data - a data structure containing the input, output variables
%   current_node - a integer denoting the current node being trained for
%   bmus - indices of the best matching units, for each training sample
%
% ++OUTPUT++
%   1. GSOM - the GSOM with the updated local models
%   2. err - updated error matrix
% Bo Lu
% University of Texas at Austin
% Nov, 2014
function [gsom, err] = netNodeRecursive_localPLS(gsom,update_data,current_node)
A = gsom.model_parameters.num_components;
model = gsom.models{current_node};

% update model parameters
model = pls_recursive(model,xnew,ynew,lambda);
% update activation
% update error
end