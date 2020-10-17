function obj = netInsert(p, q, gsom)
                % insert a node between p and q in the network and connect the newly added
                % node to p and q's common neighbors this can be easily
                % implemented by modifying the adcacency matrix. See also netSplit()
                % q, p      node indices
                % gsom      network
                obj = gsom;
                if p == q
                    warning('The nodes are the same.')
                    return
                end

                obj.netsize = gsom.netsize + 1;
                % ++ New node is inserted at the average distance point
                % between the existing two nodes
                obj.codebook = [gsom.codebook; gsom.codebook(p,:)/2 + gsom.codebook(q,:) / 2];
                
                % ++ the new RMS at the new node is taken as the average of
                %   the two existing nodes
                obj.rmsReg = [gsom.rmsReg; gsom.rmsReg(p,:)/2 + gsom.rmsReg(q,:) / 2];
                
                % ++ Activation is the number of samples belonging to this
                %       node
                obj.activation = [gsom.activation; zeros(1, size(gsom.activation, 2))]; %just append a zero. It will be updated anyway
                
                % ++ These are the beta coefficients of the RLS
                
%                obj.freq = [gsom.freq,0];

                % ++ constructing new adjacency matrix
                tempMtx = false(obj.netsize);% ++ start with no neighbors
                tempMtx(1:gsom.netsize, 1:gsom.netsize) = gsom.adjMtx; % ++ fill in the previous AdjMtx

                %delete the connection between p and q
                tempMtx(p, q) = false;
                tempMtx(q, p) = false;
                
                %find the common neighbors of node p and node q
                Ind = find(and(gsom.adjMtx(p, :),  gsom.adjMtx(q, :)) == 1);
                Ind = [Ind, p, q]; % connect p and q with newly added node also
                %connect them to the newly added node
                for i = 1:length(Ind)
                    tempMtx(Ind(i), obj.netsize) = true;
                    tempMtx(obj.netsize, Ind(i)) = true;
                end
                obj.adjMtx = sparse(tempMtx);
                %obj.adjList= adjmtx2list(obj.adjMtx);
            end