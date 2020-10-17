% GSPLS update
%
% Performs update on an already trained GSPLS model
% For simplicity purposes, this update scheme is not recursive and requires
% the use of the original training_data
%
% In the future, the update should be **automatic**.
%
% Debug mode
% ---------------------------------------
%  see debug for returned results
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

function [obj, training_data] = gspls_update_static(gsom, training_data, update_data, parameters,debugmode)
% By default, debugmode is OFF.
if nargin < 5;debugmode = 0;end

% Same parameters from the regular old training
CBTol = parameters.CBTol; %codebook tolerance
RMSTol = parameters.RMSTol; % 1 percent change at least
ErrTol= parameters.ErrTol;
sf = parameters.sf;
s0 = parameters.s0;
passMax = parameters.passMax;
LDMWidth = parameters.LDMWidth;
d2Tol = parameters.d2Tol; % maximum distance away from codebook allowed
% 1. Check if GSOM has been trained
if isempty(gsom.codebook)
    error('The GSOM is not initialzed. Please use gspls_train to first initialize the GSOM.');
end

% check for input inconsistencies
if (size(training_data.model_inputs,2) ~= size(update_data.model_inputs,2)) ...
        || (size(training_data.gsom_inputs,2) ~= size(update_data.gsom_inputs,2)) ...
        || (size(training_data.outputs,2) ~= size(update_data.outputs,2))
    error('Update data inconsistent with initial training data.');
end

% calculate d2Tol automatically by using the mean radius between the
% two clusters
DistMtx = zeros(gsom.netsize);
for i = 1:size(gsom.codebook,2);
    t = repmat(gsom.codebook(:,i),1,size(gsom.codebook,1));
    s = repmat(gsom.codebook(:,i)',size(gsom.codebook,1),1);
    DistMtx = DistMtx + (t - s).^2;
end

DistMtx = sqrt(DistMtx);
DistList = reshape(DistMtx,[],1);
DistList(DistList==0) = [];
DistList = unique(DistList);
d2Tol = mean(DistList)/2;


% 2. Store the updated training data in the training data file.

training_data.model_inputs = [training_data.model_inputs;update_data.model_inputs];
training_data.gsom_inputs = [training_data.gsom_inputs;update_data.gsom_inputs];
training_data.outputs= [training_data.outputs;update_data.outputs];


% 3. Update the GSOM iteratively
% just some dummy variables to start the loop
[bmus,d2] = netBmus(gsom.codebook, training_data, 1);
new_bmus = zeros(size(bmus));


% check to see if we need to add new nodes?
nodes = find(d2 > d2Tol);
if ~isempty(nodes)
    display('Distance exceeded tolerance, growing a new node...');
    node_center = mean(training_data.gsom_inputs(nodes,:),1);
    mask = ones(size(gsom.codebook,2),1);
    
    % calculate the distance from the node_center to the all the codebooks
    Dist = (gsom.codebook.^2)* mask * 1 ...
        + ones(gsom.netsize,1)*mask'*(node_center'.^2) ...
        - 2*gsom.codebook*diag(mask)*node_center';
    
    %return the best matching nodes
    [d2, p] = min(Dist', [], 2);
    
    % find the neighbors of p that is closest to node centroid
    Eudis1 = sum((repmat( node_center,gsom.netsize, 1) - gsom.codebook)'.^2); % squared Eudis
    Eudis1 = Eudis1 .* gsom.adjMtx(p, :);  % only search in node p's neighbors
    
    q = find(Eudis1 == min(Eudis1(Eudis1 ~= 0))); % find the index of the minimum non-zero node
    
    % insert new node at node_center and connect it to p and q.
    gsom = netAppend(node_center, p, q, gsom);
end


% are they the same? This evaluates to true the first iteration
while any(bmus ~= new_bmus)
    % get the new BMUs
    [bmus,d2] = netBmus(gsom.codebook, training_data, 1);
    err = zeros(size(training_data.outputs));
    
    % Update the local model one node at a time
    centroids = [];
    for b = 1:gsom.netsize;
        if ~any(bmus == b)
            centroids(b,:) = gsom.codebook(b,:);
            warning(sprintf('Node %d has zero activation, skipping local model fitting...',b));
            continue;
        end
        
        ind = find(bmus == b);
        
        Distance(b,:)= bfs_mtx(gsom.adjMtx,b);
        centroids(b,:) = mean(training_data.gsom_inputs(ind,:));
        
        %++ 1. just simple local geometric fit
        if strcmp(gsom.model_type,'geometric')
            [gsom,temp_err] = netNodeFit_geometric(gsom,training_data,b,bmus,d2);
        elseif strcmp(gsom.model_type,'localPCA')
            %++ 2. apply local PCA model
            [gsom,temp_err] = netNodeFit_localPCA(gsom,training_data,b,bmus);
        elseif strcmp(gsom.model_type,'coopPCA')
            %++ 3. apply cooperative PCA model
            [gsom,temp_err] = netNodeFit_coopPCA(gsom,training_data,b,bmus,Distance(b,:),1);
        elseif strcmp(gsom.model_type,'localPLS')
            %++ 4. apply local PLS model
            [gsom,temp_err] = netNodeFit_localPLS(gsom,training_data,b,bmus);
        elseif strcmp(gsom.model_type,'coopPLS')
            %++ 5. apply cooperative PLS model
            [gsom,temp_err] = netNodeFit_coopPLS(gsom,training_data,b,bmus,Distance(b,:),1);
        end
        %[gsom,y_hat,temp_err] = pwa_wls_mimo(gsom,b,bmus,X,Y,Distance(b,:),coop);
        err(bmus==b) = temp_err;
    end
    gsom.rms = sqrt(mean(err));
    
    % check if the centroids needs to be updated
    if any((gsom.codebook - centroids)>= ErrTol)
        gsom.codebook = centroids;
    end
    % we also udpate the new BMUs to reflect new trained models
    [new_bmus,d2] = netBmus(gsom.codebook, training_data, 1);
    % if the codebook hasn't been updated, then this new_bmus will equal to
    % bmus, result in loop termination
end

% so now we have convergence in codebook location AND BMUs
obj = gsom;
obj.type = 'gating';
display('Update complete');
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