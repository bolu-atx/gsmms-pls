%++ Breath-First Search based on Adjagency Matrix
function Distance= bfs_mtx(adjMtx,Source)
    %implement the breath-first search algorithm based on adjacency matrix
    %Source         the source vertice
    %Distance       the distance on the graph
    
    %++ Input: adjMtx - a square matrix of logicals (1,0)
    %          Source - a index denoting which node is of interest
    %++ Output: Distance - a matrix denoting the distance from the Source
    %               to all its neighbours

    NumofVertices=size(adjMtx);
    for p=1:NumofVertices
        Color(1,p)=0;
        Distance(1,p)=inf;
    end

    Color(1,Source)=1;
    Distance(Source)=0;
    Queue=[];

    Queue=[Queue;Source];

    while (size(Queue)>0)
        node=Queue(1,1);
        Queue(1,:)=[];
        for p=1:NumofVertices
            if (adjMtx(node,p)==1)&&(Color(1,p)==0)   
                %display(p);
                Color(1,p)=1;
                Distance(1,p)=Distance(1,node)+1;
                Dis=Distance(1,p);
                Queue=[Queue;p];
            end
        end
        Color(1,node)=2;
    end
end