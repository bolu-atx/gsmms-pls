% gspls_predict_batch.m
%
% given a trained GSOM model, predict using testing data input in batch
% (predicts all of the training test data at once without making
% modification to the GSOM)
%
%
% Returns 3 objects
%  bmus - an array of best matching units according to the labeling of GSOM
%  y_pred - a time series of predictions
%  stats - other information that might be useful for analysis, such as R2,
%           MAPE and other stats
% 
% Bo Lu
% Sept 2nd, 2014

function [bmus, y_pred, stats]= gspls_predict_batch(gsom, testing_data, parameters)
% 1. Match testing data to the nodes
% 1a calculate the GSOM inputs

% find best matching units
[bmus, d2] = netBmus(gsom.codebook,testing_data,1);

% make predictions by invoking the corresponding model for each node
y_pred = nan(size(testing_data.outputs));
stats = struct();
stats.SPE = nan(size(testing_data.model_inputs,1),1);
stats.T2 = nan(size(testing_data.model_inputs,1),1);
stats.d2 = d2;

for k = 1:gsom.netsize;
    ind = find(bmus == k); % find the indices belonging to this node
    
    % get the model corresponding to this node
    model = gsom.models{k};
    
    % make prediction
    if strcmp(gsom.model_type,'geometric')
            y_pred = [];
            stats = [];
            display('Geometric GSOM models do not support prediction.');
            display('Only the BMUs and the D2 will be returned.')
            return;
        elseif (strcmp(gsom.model_type,'localPCA') || strcmp(gsom.model_type,'coopPCA'))
            % PCA does not make a prediction atm, only return diagnostics
            % such as T2 and SPE
            [T2,SPE] = netNodePredict_PCA(model,testing_data,ind);
            y_pred = [];
            
            stats.SPE(ind,:) = SPE;
            stats.T2(ind,:) = T2;
            display('PCA GSOM models currently only support fault detection.');
            display('Only T2 and SPE will be predicted');
        elseif (strcmp(gsom.model_type,'localPLS') || strcmp(gsom.model_type,'coopPLS'))
            [y_pred_ind, stats_ind] = netNodePredict_PLS(model,testing_data, ind);
            y_pred(ind,:) = y_pred_ind;
            stats.SPE(ind,:) = stats_ind.SPE;
            stats.T2(ind,:) = stats_ind.T2;
    end
    
end
y = testing_data.outputs;
stats.residual = y_pred - y;

% calculate R2
SST=sum((y-mean(y)).^2);
SSR=sum((y_pred-mean(y)).^2);
SSE=sum((y-y_pred).^2);

stats.R2=1-SSE/SST;


stats.RMSEP = rms(stats.residual);

end

function [] = netNodeValidate_PLS(gsom, testing_data, ind);

end

function [y_pred, stats] = netNodePredict_PLS(model,testing_data, ind);
model_inputs = testing_data.model_inputs(ind,:);
model_outputs = testing_data.outputs(ind,:);
[y_pred, RMSEP, diagnostics] = plsval(model,model_inputs,model_outputs);
stats = diagnostics;
stats.RMSEP = RMSEP;
end

function [y_pred, stats] = netNodePredict_coopPLS(model,testing_data, ind);

end

% T2, SPE are returned by the netNodePredict_PCA
% Finds the best matching unit and then returns the
function [T2, SPE] = netNodePredict_PCA(model,testing_data, ind);
model_inputs = testing_data.model_inputs(ind,:);
model_outputs = testing_data.outputs(ind,:);
[y_pred, RMSEP, diagnostics] = plsval(model,model_inputs,model_outputs);
stats = diagnostics;
stats.RMSEP = RMSEP;
end