% plot_colorby function  plot_colorby(x,y,s,colorby)
%
% same functionality as regular plot(), except that it will color the
% entries by color according to the class label vector colorby
%
% colorby can be in two forms: 
%  1. a vector of numerical class labels.
%  2. a cell array of string class labels.
%
% Example usage:
%   plot_colorby(scores(:,1),scores(:,2),'o',RS.lot_id);
%
%    This will color the score plot by the lot_id of the data set
% 
% Bo Lu, UT-Austin
% March 12th, 2014

function plot_colorby(x,y,s,colorby)
if nargin == 3;colorby = s;s = y;y = x; x= 1:numel(y);end;
if numel(x) ~= numel(y) || numel(colorby) ~= numel(x);error('Dimensions inconsistency.');end
%h = plot(x,y,s);

unique_ids = unique(colorby);
cmap = hsv(numel(unique_ids ));  %# Creates a 6-by-3 set of colors from the HSV colormap

for i = 1:numel(unique_ids)
    current_id = unique_ids(i);
    if strcmp(class(colorby),'cell')
        current_rows = find(ismember(colorby,current_id));
    else
        current_rows = find(colorby == current_id);
    end
    
    hold on;plot(x(current_rows),y(current_rows),s,'Color',cmap(i,:));
end


if strcmp(class(colorby),'cell')
    legend(unique_ids);
% else
%     legend(cellstr(num2str(unique_ids')))
end
end