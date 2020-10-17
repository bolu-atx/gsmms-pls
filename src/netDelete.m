% netDelete.m
% ----------------------------------------------------------------------------
% deletes a node p from the current network structure and then plots the
% resulting network after node deletion

% Simple step to delete node:
% 1. Look all for all nodes connected with p
% 2. Connect the neighbors of they share the connection to p (which is to
%    be deleted)
%
% Using this scheme guarantees that the simplified GSOM will be of the same
% structure as before.
%
%  Bo LU, Oct 20th, 2014
%  The University of Texas-Austin

function obj = netDelete(p, gsom)
obj = gsom;

% shrink the network

obj.netsize = gsom.netsize - 1;

% Find the neighbors of p first
neighbors = find(obj.adjMtx(p,:)==1);

% copy the adjMtx to make adjustments
tempMtx = obj.adjMtx;

% loop through all the neighbors connected
for i = 1:numel(neighbors)
    q = neighbors(i);
    not_q = setdiff(neighbors,q);
    
    % connect q to all the neighbors of p.
    tempMtx(q,not_q) = 1;
    
    % don't worry about symmetry since we loop through all
    % the neighbors
end


% remove the pth node completely from the AdjMtx
tempMtx(p,:) = [];
tempMtx(:,p) = [];

obj.adjMtx = tempMtx;

% remove other stats associated with this node
obj.rmsReg(p) = [];
obj.activation(p) = [];
obj.models(p) = [];
obj.codebook(p,:) = [];
end