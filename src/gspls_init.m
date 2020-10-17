% GSPLS - initialize the SOM data structure
function obj = gspls_init(varargin) %                           
if nargin==0  
    obj.model_info = [];
    obj.model_info.comments = [];
    % these variables are specified later during training
    obj.netsize = []; % size of the self organizing map
    obj.codebook = []; % centroid coordinates for each node
    obj.rmsReg = []; % root mean squared error regularized
    obj.activation = []; % number of times each node has been activated
    obj.models = {}; % store the local models
    obj.rms = []; % root mean squared error
    
    obj.adjMtx = [];
    obj.sigma = [];
    obj.type = []; % is 'Gaussian' or 'Piecewise' for now

elseif isa(varargin,'GSPLS')
    obj = varargin;
end