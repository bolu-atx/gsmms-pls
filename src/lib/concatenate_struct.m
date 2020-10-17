% concatcnates structures elementwise
function c = concatenate_struct(a,b)
fields = fieldnames(a);
c = a;
try
    for i = 1:size(fields);
        field = char(fields(i));
        c.(field) = [a.(field);b.(field)];
    end
catch
    warning('Concatenation failed, returning first struct.');
    c = a;
end
end