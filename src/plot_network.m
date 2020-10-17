% Visualize the GSOM network
% P stands for
function plot_network(MMS,training_data,bmus,titlestring,p,q)
    clf
    hold on
    if size(training_data.gsom_inputs,1)>1000;
        plot_colorby(training_data.gsom_inputs(1:5:end,1),training_data.gsom_inputs(1:5:end,2),'+',bmus(1:5:end));
    else
        plot_colorby(training_data.gsom_inputs(1:2:end,1),training_data.gsom_inputs(1:2:end,2),'+',bmus(1:2:end));
    end
    if size(MMS.codebook,2)==1
        plot(MMS.codebook(:, 1),0, '.', 'markersize', 30); 
    else    
        for i = 1:MMS.netsize-1
            for j = i+1:MMS.netsize
                if MMS.adjMtx(i, j) == 1
                    plot(MMS.codebook([i, j], 1), MMS.codebook([i, j], end));
                end
            end
        end
        plot(MMS.codebook(:, 1), MMS.codebook(:, end), '.', 'markersize', 30);
        for i = 1:MMS.netsize
            text(MMS.codebook(i, 1), MMS.codebook(i, end), sprintf('\\color{red} Node %d',i));
        end
    end
    title(titlestring);
    if nargin == 5
        % p is the max error node
        plot(MMS.codebook(p, 1),MMS.codebook(p, end), 'r.', 'markersize', 30); 
        % q is the furthest away node
        plot(MMS.codebook(q, 1),MMS.codebook(q, end), 'ro', 'markersize', 15); 
    end
    hold off
    grid on
    drawnow
    
end