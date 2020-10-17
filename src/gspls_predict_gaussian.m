% gspls_predict_gaussian.m
%
% Applies weighted prediction depending on the distances away from each
% node.
%
% Returns 3 objects
%  bmus - an array of best matching units according to the labeling of GSOM
%  y_pred - a time series of predictions
%  stats - other information that might be useful for analysis, such as R2,
%           MAPE and other stats
% 
% Bo Lu
% Sept 2nd, 2014
%
function [bmus, y_pred, stats]= gspls_predict_gaussian(gsom, testing_data, parameters)
% avg distance is the average distance among the codebook centroids
% we need this value to scale the weight vectors assigned to each local
% model
avg_dist = mean(mean(compute_distances(gsom.codebook,gsom.codebook)));

% preallocate some memories for the outputs
y_pred = zeros(size(testing_data.outputs));
stats = struct();
stats.SPE = nan(size(testing_data.model_inputs,1),1);
stats.T2 = nan(size(testing_data.model_inputs,1),1);


% Calculate the distance of testing samples from the GSOM centroids
% d is NETSIZE x N_testing_sample
d = compute_distances(gsom.codebook,testing_data.gsom_inputs);
[wtf,bmus] = min(d); % wtf is useless
wj = zeros(size(d));

% we use the negative exponential weighting scheme
% the tuning parameter is LDMWidth, 0 for no impact, 1 for a lot of
% cooperation
for i = 1:gsom.netsize
    wj(i,:) = exp(-d(i,:).^2/(2*(avg_dist^2*(parameters.LDMWidth+0.01))));
end

w = wj./repmat(sum(wj),gsom.netsize,1); % weights to be applied to predictions
w = w'; % transpose into column vector
hist(reshape(w,[],1)); % for debug
stats.w = w;

% make prediction using each local node and combine it using the weight
% vectors calculated above.
ind = 1:size(testing_data.outputs,1);
for k = 1:gsom.netsize;
    % legacy construct, so here we set it to predicting everything

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
            y_pred(ind,:) = y_pred(ind,:)+ y_pred_ind.*w(:,k); %
            stats.SPE(ind,:) = stats.SPE(ind,:) + stats_ind.SPE.*w(:,k);
            stats.T2(ind,:) = stats.T2(ind,:) + stats_ind.T2.*w(:,k);
    end
    
end
% real Y, using this to calculate R2 and RMSEP
y = testing_data.outputs;
stats.residual = y_pred - y;

% calculate R2
SST=sum((y-mean(y)).^2);
SSR=sum((y_pred-mean(y)).^2);
SSE=sum((y-y_pred).^2);

stats.R2=1-SSE/SST;
stats.RMSEP = rms(stats.residual);
plot_network(gsom,testing_data,bmus,sprintf('Current network with testing input mapped'));
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