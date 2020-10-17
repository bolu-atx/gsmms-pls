%% Setup the training dataset.
% training_data.inputs = scores(:,1:2);
% training_data.outputs = y;


training_data = struct();
training_data.model_inputs = Xs_centered;
training_data.gsom_inputs = T(:,1:3);
training_data.outputs = y;



testing_data = struct();
testing_data.model_inputs = T;
testing_data.gsom_inputs = simulated_data_set.gsom_inputs(testing,1:2);
testing_data.outputs = simulated_data_set.outputs(testing,:);
testing_data.true_mode = simulated_data_set.true_mode(testing,:);
testing_data.true_betas = simulated_data_set.true_betas;

%% Setup the training parameters
parameters = struct();
parameters.CBTol = 1e-4; %codebook delta tolerance
parameters.sf = 0.01; % cooperation parameters
                      % alpha = -log(sf/s0)/(passMax-1);
                      % Sigma = s0*exp(-alpha*((1:passMax)-1));
parameters.s0 = 1;
parameters.passMax = 50; % max number of passes for GSOM update
parameters.LDMWidth = 0; % tuning parameter, 1 for pure cooperative, 0 for no cooperation
parameters.gsom_netSize = 3; % initial starting GSOM size
parameters.ErrTol = 1e-5; % error tolerance
parameters.max_netSize = 10;
parameters.RMSTol=0.01; % 1 percent change at least
% uncomment to enable writing animation
parameters.gifName = 'simul.gif';
parameters.d2Tol = 20; % distance tolerance for adapting new nodes

%% Initialize GSOM
gsom = gspls_init();
gsom.model_info.comments = 'Testing GSOM';
% gsom.model_type = 'geometric'; % use simple k-means clustering
% gsom.model_type = 'localPCA'; % apply local PCA
% gsom.model_type = 'coopPCA'; % apply cooperative PCA
% gsom.model_type = 'coopPLS'; % apply coop PLS
gsom.model_type = 'localPLS'; % apply local PLS
gsom.model_parameters = struct();
gsom.model_parameters.num_components = 5;

%% Train GSOM
dbstop if error
[gsom, debug] = gspls_train(gsom, training_data, parameters,1);


figure;
subplot(211);bar(gsom.activation);ylabel('# of Activations');
subplot(212);bar(gsom.rmsReg);ylabel('Root-mean-squared Error');
xlabel('Mode')

%% Update GSOM
update_data = testing_data;
[gsom_updated, training_data_updated] = gspls_update_static(gsom,training_data,update_data,parameters,0);


%% Predict using GSOM

% batch predict mode
[bmus, y_pred, stats]= gspls_predict_batch(gsom, testing_data, parameters);
[bmus2, y_pred2, stats2]= gspls_predict_gaussian(gsom, testing_data, parameters);


%% Figure 1. Predicted vs Measured
figure;
plot(y_pred2,'-b');
hold on;
plot_colorby(1:numel(testing_data.outputs),testing_data.outputs,'o',bmus);
title(sprintf('R^2 = %0.3f',stats.R2));
legend('Predicted','Measured');


%%
% live online predict
%
% [bmus, y_pred, stats]= gspls_predict_live(gsom, testing_data, parameters);


%%
% train for an simulation lower bound first

model = pls(training_data.model_inputs,training_data.outputs,5);
[ypred, rmsep, diag] = plsval(model,testing_data.model_inputs,testing_data.outputs,5);
worst_case.R2 = diag.R2;
worst_case.rmsep = rmsep


%%
% best case scenario
%

ypred = nan(size(testing_data.outputs));
ytrue = nan(size(testing_data.outputs));

for mode = 1:max(training_data.true_mode);
    ind = find(training_data.true_mode == mode);
    model = pls(training_data.model_inputs(ind,:),training_data.outputs(ind,:),5);
    
    ind_test = find(testing_data.true_mode == mode);
    ypred_mode = plsval(model,testing_data.model_inputs(ind_test,:),testing_data.outputs(ind_test,:),5);
    ytrue_mode = testing_data.outputs(ind_test,:);
    
    mode
    SST=sum((ytrue_mode-mean(ytrue_mode)).^2);
    SSR=sum((ypred_mode-mean(ytrue_mode)).^2);
    SSE=sum((ytrue_mode-ypred_mode).^2);
    R2=1-SSE/SST
    
    ypred(ind_test,:) = ypred_mode;
    ytrue(ind_test,:) = ytrue_mode;
end

SST=sum((ytrue-mean(ytrue)).^2);
SSR=sum((ypred-mean(ytrue)).^2);
SSE=sum((ytrue-ypred).^2);
R2=1-SSE/SST;
rmsep=sqrt(sum((ypred-ytrue).^2)/length(ytrue));


best_case.R2 = R2;
best_case.rmsep = rmsep;

%% output
stats
stats2
best_case
worst_case
%% debug
% close all
% mode = 3
% 
% ind = find(training_data.true_mode == mode);
% ind_t = find(testing_data.true_mode == mode);
% 
% model = pls(training_data.model_inputs(ind,:),training_data.outputs(ind,:),5);
% 
% subplot(211);
% plot(plsval(model,training_data.model_inputs(ind,:),training_data.outputs(ind,:)))
% hold on
% plot(training_data.outputs(ind,:),'ro');
% 
% subplot(212);
% plot(plsval(model,testing_data.model_inputs(ind_t,:),testing_data.outputs(ind_t,:)))
% hold on;
% plot(testing_data.outputs(ind_t),'ro');