% function [gsom, y_hat, err] = netNodeFit_geometric(gsom,training_data,current_node,bmus,top_dis,coop)
% Determine the error of each net based purely on distances.

function [gsom, err] = netNodeFit_geometric(gsom,training_data,current_node,bmus,d2)
    % Since we already calculated the BMUs
    % we can just use the distances calculated there here)
    err = sqrt(d2(bmus == current_node));
    gsom.rmsReg(current_node) = sqrt(mean( sum(err.^2,2) ));
    gsom.activation(current_node) = size(err,1);
end