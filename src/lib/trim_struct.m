% trim_struct
function c = trim_struct(a, idx)
fields = fieldnames(a);
c = a;
    for i = 1:size(fields);
        try
        field = char(fields(i));
        c.(field)(idx,:) = [];
        catch
            %do nothing
        end
    end   
end