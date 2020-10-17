% GSOM validation function
% given a trained GSOM model, predict using testing data input
%
% Returns 3 objects
%  bmus - an array of best matching units according to the labeling of GSOM
%  y_pred - a time series of predictions
%  stats - other information that might be useful for analysis, such as R2,
%           MAPE and other stats
% 
% Bo Lu
% Sept 2nd, 2014

function [bmus, y_pred, stats]= gspls_predict(gsom, testing_data, parameters)
% 1. Match testing data to the nodes
% 1a calculate the GSOM inputs

% find best matching units
[bmus, d2] = netBmus(gsom.codebook,testing_data,1);


% make prediction for each BMU
for k = 1:gsom.netsize;
    ind = find(bmus == k);
    model = gsom.models{k};
    [T2, SPE] = netNodeValidate_PCA(model,testing_data,ind);
    [T2, SPE] = netNodeValidate_PLS(model,testing_data,ind);
    T2(ind) = T2;
    SPE(ind) = SPE;
end

plot_network(gsom,training_data,sprintf('Current network'));
[nr nc] = size(testing_data.inputs);

end

function [] = netNodeValidate_PLS(gsom, testing_data, ind);

end

function [] = netNodePredict_PLS(gsom, testing_data, ind);
end