function d = compute_distances(codebook, input_vector,mask)
    % calculate the best matching units
    % codebook           the codebook. each row is a codeword
    % input_vector       the input_vectors, each row is a data vector
    % n                  if n = 1 only calculate bmu, n =2 calculate the first
    %                    two bmus and so on
    % mask               same dimension with codeword; default [1 1 ... 1]' 

    N = size(input_vector, 1); %number of samples
    [netsize, dim] = size(codebook);

    if nargin < 3
        mask = ones(dim, 1);
    end

    d = sqrt((codebook.^2)* mask * ones(1,N) ... 
        + ones(netsize,1)*mask'*(input_vector'.^2) ... 
        - 2*codebook*diag(mask)*input_vector'); 
    
    %++ Distance = SQRT(CODEBOOK^2 + INPUT^2 - 2*CODEBOOK*INPUT)