% ++ BMUs is a index vector, size is N x P, N is number of rows in input
%       vector, P is the number of codewords in codebook
function [bmus, quanErrors] = netBmus(codebook, training_data, n)
% calculate the best matching units
% codebook           the codebook. each row is a codeword
% training_data.inputs       the input_vectors, each row is a data vector
% n                  if n = 1 only calculate bmu, n =2 calculate the first
%                    two bmus and so on
% mask               same dimension with codeword; default [1 1 ... 1]'

N = size(training_data.gsom_inputs, 1); %number of samples
[netsize, dim] = size(codebook);
inputs = training_data.gsom_inputs;
mask = ones(dim,1);
% calculation can be made much faster by transforming it to a matrix operation:
%   Dist = (codebook.^2)* mask * ones(1,N)
%         + ones(netsize,1)*mask'*(training_data.inputs'.^2)
%         - 2*codebook*diag(mask)*training_data.inputs'
% ++ a simplified way to calculate the distance of each
% sample vector entry to the codeword nodes
%++ N x P dimension, where P is the number of codeword nodes
% Dist = (codebook.^2)* mask * ones(1,N) ...
%     + ones(netsize,1)*mask'*(inputs'.^2) ...
%     - 2*codebook*diag(mask)*inputs';

 Dist = (codebook.^2)* mask * ones(1,N) ...
     + ones(netsize,1)*mask'*(inputs'.^2) ...
     - 2*codebook*diag(mask)*inputs';

if n == 1
    %++ returns a index list of the best matching node
    [quanErrors, bmus] = min(Dist', [], 2);
elseif n > 1
    [quanErrors, bmus] = sort(Dist', 2);
    quanErrors = quanErrors(:, 1:n);
    bmus = bmus(:, 1:n);
end
quanErrors = sqrt(quanErrors);
end
