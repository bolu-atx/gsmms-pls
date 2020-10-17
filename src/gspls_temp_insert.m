% temporary function to insert a node on GSOM
% k the ref node
%
function gsom_u = gspls_temp_insert(gsom,k,offset)
codebook = gsom.codebook;
new_loc = codebook(k,:) + offset;

netsize = size(gsom.adjMtx,1); % old net size
m = netsize+1;

adjMtx = sparse(zeros(size(gsom.adjMtx)+1));
adjMtx(1:netsize,1:netsize) = gsom.adjMtx;

prim_neighbors = find(adjMtx(k,:));
% connect the new node
for j = [prim_neighbors k];
   adjMtx(j,m) = 1;
   adjMtx(m,j) = 1;
end

% set its codebook location
codebook = [codebook;new_loc];

gsom_u = gsom;
gsom_u.codebook = codebook;
gsom_u.adjMtx = adjMtx;
gsom_u.netsize = netsize+1;
gsom_u.rmsReg = [gsom.rmsReg;0];
gsom_u.activation = [gsom.activation;0];
gsom_u.models{m} = gsom.models{k};
end