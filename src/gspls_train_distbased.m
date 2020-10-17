% GSPLS_train
%
% Trains the GSPLS model to obtain a SOM and the submodels associated with
% each node. Currently only works in batch mode with piece-wise linear models
%
% Returns an GSOM object
% For the detailed description of the GSOM object, see gsom_init.m
%
% Paper Reference:
% Jianbo Liu, Dragan Djurdjanovic, Kenneth Marko, and Jun Ni, Growing
% Structure Multiple Model Systems for Anomaly Detection and Fault
% Diagnosis, Journal of Dynamic Systems, Measurement and Control, Sept.
% 2009 Vol 131
%
% Bo Lu, UT-Austin |  July, 2014
% bo.lu@utexas.edu

function [obj, debug] = gspls_train(gsom, training_data, parameters)
CBTol = parameters.CBTol;
RMSTol = parameters.RMSTol; % 1 percent change at least
sf = parameters.sf;
s0 = parameters.s0;
passMax = parameters.passMax;
LDMWidth = parameters.LDMWidth;

% if we have a gifName set in the object, then enable writing gif animation
if isfield(parameters,'gifName')
    filename = parameters.gifName;
    makegif = 1;
    if exist(filename,'file')
        delete(filename);
    end
else
    makegif = 0;
end


if isempty(gsom.netsize)
    gsom.netsize = parameters.gsom_netSize;
end
error_tolerance = parameters.error_tolerance;
max_netSize = parameters.max_netSize;

% Cooperation parameters
alpha = -log(sf/s0)/(passMax-1);
Sigma = s0*exp(-alpha*((1:passMax)-1));

%initialize with all zeros
gsom.activation = zeros(gsom.netsize,1);
gsom.rmsReg = zeros(gsom.netsize,1);

figure;
%% gsom = netInit(Z, gsom, 'random'); % initialize the codebook
% initizliaze the codebook
% case 1. K-means initialization
% [idx, gsom.codebook] = kmeans(training_data.inputs, gsom.netsize);
% case 2. random initialization
if isempty(gsom.codebook)
    gsom.codebook = training_data.gsom_inputs(ceil(rand(gsom.netsize, 1) * ...
        size(training_data.gsom_inputs, 1)), :);
    
    gsom.adjMtx = true(gsom.netsize, gsom.netsize);
    
    % make the diagonal false
    for i = 1:gsom.netsize
        gsom.adjMtx(i, i) = false;
    end
    
    
    %+ ----------- Piecewise Training --------------------------
    % First, initialize the GSOM
    %
    % Calculate the best matching units and their distance from the center
    [bmus, d2] = netBmus(gsom.codebook,training_data,1);
    
    % Initializations of the local model parameters
    Distance = zeros(gsom.netsize);
    err = zeros(size(training_data.outputs));
    
    % ++ loop through each node
    for b = 1:gsom.netsize
        % empty BMUs
        if ~any(bmus == b)
            warning('Zero activation.')
            continue;
        end
        %++ calculate its distance to the neighbors
        Distance(b,:)= bfs_mtx(gsom.adjMtx,b); % topological distance
        % calculate the actual distance as well
        d = compute_distances(gsom.codebook,training_data.gsom_inputs);
        
        %++ simple local geometric clustering
        if strcmp(gsom.model_type,'geometric')
            [gsom,temp_err] = netNodeFit_geometric(gsom,training_data,b,bmus,d2);
        elseif strcmp(gsom.model_type,'localPCA')
            %++ local PCA model
            [gsom,temp_err] = netNodeFit_localPCA(gsom,training_data,b,bmus);
        elseif strcmp(gsom.model_type,'coopPCA')
            %++ cooperative PCA model
            [gsom,temp_err] = netNodeFit_coopPCA(gsom,training_data,b,bmus,Distance(b,:),1);
        elseif strcmp(gsom.model_type,'localPLS')
            %++ apply local PLS model
            [gsom,temp_err] = netNodeFit_localPLS(gsom,training_data,b,bmus);
        elseif strcmp(gsom.model_type,'coopPLS')
            %++ apply cooperative PLS model
            [gsom,temp_err] = netNodeFit_coopPLS(gsom,training_data,b,bmus,d(b,:),parameters.LDMWidth);
        end
        
        %++ update the local error
        err(bmus==b) = temp_err;
    end
    
    %++ update the global RMS for the entire system
    gsom.rms = sqrt(mean(err));
end
rms_history = gsom.rms;
bmus_old = zeros(size(training_data.gsom_inputs,1),1);

% Main loop to update the GSOM
while(gsom.rms >= error_tolerance) && (gsom.netsize < max_netSize) ...
        && ~isnan(gsom.rms)
    %estimate local model parameters and structural parameters
    disp(['Netsize = ',num2str(gsom.netsize)])

    % ++ why are we doing 200 passes?
    %  Sigma is changing every pass - gradually reduced
    %
    for pass = 1 : 1 : passMax
        codebookOld = gsom.codebook;
        coop = Sigma(pass);
        
        % update the animation file every 5 passes
        if (mod(pass,5) == 1) && (makegif == 1);
            writegif(filename,1);
        end
        
        % find the Best Matching unit for each training sample
        [bmus,d2] = netBmus(gsom.codebook, training_data, 1);
        plot_network(gsom,training_data,bmus,sprintf('Pass = %d',pass));
        % Only re-train the network when the BMUs are changed
        
        if any(bmus ~= bmus_old)
            % plot network for training visualization
            Distance = zeros(gsom.netsize);
            %++calculate the current error of the network before updating the
            % structural parameters
            for b = 1:gsom.netsize
                if ~any(bmus == b)
                    warning('Zero activation.')
                    continue;
                end
                
                Distance(b,:)= bfs_mtx(gsom.adjMtx,b); % topological distance
                %++ 1. just simple local geometric fit
                if strcmp(gsom.model_type,'geometric')
                    [gsom,temp_err] = netNodeFit_geometric(gsom,training_data,b,bmus,d2);
                elseif strcmp(gsom.model_type,'localPCA')
                    %++ 2. apply local PCA model
                    [gsom,temp_err] = netNodeFit_localPCA(gsom,training_data,b,bmus);
                elseif strcmp(gsom.model_type,'coopPCA')
                    %++ 3. apply cooperative PCA model
                    [gsom,temp_err] = netNodeFit_coopPCA(gsom,training_data,b,bmus,Distance(b,:),coop);
                elseif strcmp(gsom.model_type,'localPLS')
                    %++ 4. apply local PLS model
                    [gsom,temp_err] = netNodeFit_localPLS(gsom,training_data,b,bmus);
                elseif strcmp(gsom.model_type,'coopPLS')
                    %++ 5. apply cooperative PLS model
                    [gsom,temp_err] = netNodeFit_coopPLS(gsom,training_data,b,bmus,d(b,:),parameters.LDMWidth);
                end
                %[gsom,y_hat,temp_err] = pwa_wls_mimo(gsom,b,bmus,X,Y,Distance(b,:),coop);
                err(bmus==b) = temp_err;
            end
            err = reshape(err, max(size(err)),1);
            % Update the root mean squared error
            gsom.rms = sqrt(mean(err));
            
            % if the maximum mean fitting error is small then go to fine tuning
            if gsom.rms  < error_tolerance && ~isnan(gsom.rms)
                disp('Errors below tolerance')
                break;
            end
        else
            % if we have no changes in BMUs, we'll run into errors in the
            % next step... unless we fill in some distances here
            old_dist = Distance;
            Distance = zeros(gsom.netsize);
            Distance(1:size(old_dist,1),1:size(old_dist,2)) = old_dist;
        end
        
        %update the structural parameter (weight vectors)
        for i = 1 : gsom.netsize
            ind = find( bmus(:, 1) == i);
            
            % if we have no points belonging to this node, this becomes a special case.
            % The idea here is to have this point move to its neighbor with the highest
            % fitting error.
            if isempty(ind)
                display('zero activation!');
                %pause;
                % find the gradient direction
                neighbors = gsom.adjMtx(i,:);
                no_neighbors = sum(full(neighbors));
                h = exp(-1^2 / 2/(coop)^2 );
                % calculate the gradient
                % as the weighted average of the codebooks of the neighbors
                codebook_gradient = mean((gsom.codebook(neighbors,:)-repmat(gsom.codebook(i,:),no_neighbors,1))...
                    .*repmat(gsom.rmsReg(neighbors).^2,1,size(gsom.codebook,2))...
                    /sum(gsom.rmsReg(neighbors)));
                codebook_gradient = codebook_gradient./norm(codebook_gradient);
                learning_rate = h * mean(mean(abs(gsom.codebook)))/2
                if (~any(isnan(codebook_gradient))) && (~isnan(learning_rate))
                    gsom.codebook(i, :) = gsom.codebook(i, :) + learning_rate * codebook_gradient;
                end
                continue;
            end
            
            %calculate the distance of the nodes on the network w.r.t. ith
            %node using breath first search algorithm from the adjMtx
            for j = 1:gsom.netsize
                gravity_ratio = numel(ind)/(numel(ind)+numel(find( bmus(:, 1) == j)));
                h = exp(-Distance(i, j)^2 / 2/(coop)^2 );
                codebook_gradient = ( mean(training_data.gsom_inputs(ind, :)) - gsom.codebook(j, :));
                learning_rate = gsom.rmsReg(j, 1) / sum(gsom.rmsReg)  * h * gravity_ratio;
                if (~any(isnan(codebook_gradient))) && (~isnan(learning_rate))
                    gsom.codebook(j, :) = gsom.codebook(j, :) + learning_rate * codebook_gradient;
                end
            end
        end
        
        % break if codebook hasn't really changed
        if norm(codebookOld - gsom.codebook) < CBTol
            break
        end
        bmus_old = bmus; % save the current run BMUs
    end
    
    rms_history = [rms_history,gsom.rms]
    if ((rms_history(end-1)-rms_history(end))/rms_history(end) <= RMSTol) || (rms_history(end-1)-rms_history(end) < 0 )
        warning('RMS not improving anymore ...')
        if exist('old_gsom')
            gsom = old_gsom; % return to previous state;
            display('Deleting the last node...');
            plot_network(gsom,training_data,bmus,'Training completed.');
        end
        break;
    end

    
    %splitting if the goal is not matched by inserting several nodes at
    %a time near the node/region where we have highest fitting errors
    if ( gsom.rms > error_tolerance && gsom.netsize < max_netSize)
        old_gsom = gsom; % save the previous state
        % ++ find the maximum error node
        [maxRmsReg, p1] = max(gsom.rmsReg); % maximum RMS criteria
        netSize = gsom.netsize;
        % ++ calculate the distance of all other nodes to this node.
        Eudis1 = sum((repmat( gsom.codebook(p1, :),netSize, 1) - gsom.codebook)'.^2); % squared Eudis
        
        % ++ Mask everything else except its neighbors
        Eudis1 = Eudis1 .* gsom.adjMtx(p1, :);  % only search in node p's neighbors
        
        % ++ Finds the furthest neighbor from the current node (with
        %    highest RMS)
        [maxDis, q1] = max(Eudis1);
        
        %Insert a node between p and q. ( It is possible
        %several nodes can be added
        plot_network(gsom,training_data,bmus,'New node to be added.',p1, q1);
        %pause;
        if makegif==1
            writegif(filename ,1,2);
        end
        gsom = netInsert(p1, q1, gsom);
    end
end
obj = gsom;
obj.type = 'gating';
disp('Done!')

debug = struct();
debug.rms_history = rms_history;
debug.bmus = bmus;
end

% Visualization aid to write the progression of the training into a GIF
% file
function writegif(filename,figurehandle,framedelay)
if nargin < 3;framedelay = 0.5;end
frame = getframe(1);
im = frame2im(frame);
[imind,cm] = rgb2ind(im,256);
if exist(filename,'file')
    imwrite(imind,cm,filename,'gif','WriteMode','append','DelayTime',framedelay);
else
    imwrite(imind,cm,filename,'gif', 'Loopcount',inf);
end
end