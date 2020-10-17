% GSPLS Main
% Main script to demonstrate performing GSOM along with PLS
% reset everything

clear all;clc;close all;
addpath lib
% addpath datasets\dow
%%
rmpath datasets\dow
addpath datasets\gate_etch
load_data_gate_etch

%%
% rmpath datasets\dow
% rmpath datasets\gate_etch
% addpath datasets\metal_etch
% load_metal_etch_data

%% Setup the training parameters
parameters = struct();
parameters.CBTol = 1e-4; %codebook delta tolerance
parameters.sf = 0.01; % cooperation parameters
                      % alpha = -log(sf/s0)/(passMax-1);
                      % Sigma = s0*exp(-alpha*((1:passMax)-1));
parameters.s0 = 1;
parameters.passMax = 50;
parameters.LDMWidth = 0.1;
parameters.gsom_netSize = 3;
parameters.ErrTol = 1e-5;
parameters.max_netSize = 10;
parameters.RMSTol=0.005; % 1 percent change at least
% uncomment to enable writing animation
%parameters.gifName = 'training_local.gif';

%% Initialize GSOM
gsom = gspls_init();
gsom.model_info.comments = 'Testing GSOM';
% gsom.model_type = 'geometric'; % use simple k-means clustering
% gsom.model_type = 'localPCA'; % apply local PCA
% gsom.model_type = 'coopPCA'; % apply cooperative PCA
gsom.model_type = 'coopPLS'; % apply coop PLS
% gsom.model_type = 'localPLS'; % apply local PLS
gsom.model_parameters = struct();
gsom.model_parameters.num_components = 2;

%% Train GSOM
[gsom, debug] = gspls_train(gsom, training_data, parameters);

%% Predict using GSOM

% batch predict mode
[bmus, y_pred1, stats]= gspls_predict_batch(gsom, testing_data, parameters);
[bmus2, y_pred2, stats2]= gspls_predict_gaussian(gsom, testing_data, parameters);
figure;
plot(testing_data.outputs,'ob');hold on;plot(y_pred1,'--r');
title(sprintf('R^2 = %0.2f',stats.R2));
legend('Meas','Predicted');

%%
% live online predict
%
% [bmus, y_pred, stats]= gspls_predict_live(gsom, testing_data, parameters);

model = pls(training_data.model_inputs,training_data.outputs,5);
[y_pred3, rmsep, diag] = plsval(model,testing_data.model_inputs,testing_data.outputs,5);
worst_case.R2 = diag.R2;
worst_case.rmsep = rmsep

%% adaptive predict GSMMS
y_true = testing_data.outputs;
% update the gsom every 5 samples

starting_indices = 1:20:size(testing_data.outputs,1);
ending_indices = [starting_indices(2:end)-1 size(testing_data.outputs,1)];

gsom_u = gsom;
BMUs = [];
ypred_r = [];
close all;

for i = 1:numel(starting_indices);
    
    % format the update dataset
    update_data = struct();
    update_data.gsom_inputs = testing_data.gsom_inputs(starting_indices(i):ending_indices(i),:);
    update_data.model_inputs = testing_data.model_inputs(starting_indices(i):ending_indices(i),:);
    update_data.outputs = testing_data.outputs(starting_indices(i):ending_indices(i),:);
    
    if i == 1
        update_databank = update_data;
    else
        update_databank = concatenate_struct(update_databank,update_data);
    end
    
    
    % make prediction
    [bmus, yp_r, stats]= gspls_predict_batch(gsom_u, update_data, parameters);
    
    ypred_r = [ypred_r;yp_r];
    BMUs = [BMUs;bmus];
    % update the model
    [gsom_u,training_data,debug] = gspls_update(gsom_u, training_data, update_data, parameters,0);
    % add a new node at i = 70
    if i == 25;
        [v,k] = min(gsom.codebook(:,1));
        gsom_u = gspls_temp_insert(gsom_u,k,[-180 -400]);
    end
    %plot_network(gsom_u,update_databank,BMUs,sprintf('Current network with testing input mapped'));
    plot_colorby(1:numel(ypred_r),ypred_r,'+',BMUs);
    hold on;
    plot(1:numel(ypred_r),y_true(1:numel(ypred_r)),'--k');
     xlim([0 1200]);
     ylim([1 1.5]);
    print(['gsom_predict_trend_' num2str(i)],'-dpng');
end
clear update_databank;
%%

ymixed = (y_true + y_pred1)/2;

%% K-means
