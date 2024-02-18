% Code for the following paper:
% H. L. Shen, H. G. Zhang, S. J. Shao, and J. H. Xin, 
% Simple and Efficient Method for Specularity Removal in an Image, 

clc
clear 
close all
Image_dir = 'input';
listing = cat(1, dir(fullfile(Image_dir, '*.*G')));
% The final output will be saved in this directory:
result_dir = fullfile(Image_dir, 'result');
% Preparations for saving results.
if ~exist(result_dir, 'dir'), mkdir(result_dir); end

for i_img = 1:length(listing)
    % Input = imread(fullfile(Image_dir,listing(i_img).name));
    sfi = fix(fullfile(Image_dir,listing(i_img).name));
    imwrite(sfi, fullfile(result_dir,listing(i_img).name));
end

function [Idf] = fix(fname)
threshold_chroma = 0.03;
nu = 0.5;

I = imread(fname);
I = double(I);
[height, width, dim] = size(I);

I3c = reshape(I, height*width, 3);
tic;
% calculate specular-free image
Imin = min(I3c, [], 2);
Imax = max(I3c, [], 2);
Ithresh = mean(Imin) + nu * std(Imin);
Iss = I3c - repmat(Imin, 1, 3) .* (Imin > Ithresh) + Ithresh * (Imin > Ithresh);

% calculate specular component
IBeta = (Imin - Ithresh) .* (Imin > Ithresh) + 0;

% estimate largest region of highlight
IHighlight = reshape(IBeta, height, width, 1);
IHighlight = mat2gray(IHighlight);
IHighlight = im2bw(IHighlight, 0.1);
IDominantRegion = bwareafilt(IHighlight, 1, 'largest');

% dilate largest region by 5 pixels to obtain its surrounding region
se = strel('square',5);
ISurroundingRegion = imdilate(IDominantRegion, se);
ISurroundingRegion = logical(imabsdiff(ISurroundingRegion, IDominantRegion));

% Solve least squares problem
Vdom = mean(I3c(IDominantRegion, :));
Vsur = mean(I3c(ISurroundingRegion, :));
Betadom = mean(IBeta(IDominantRegion, :));
Betasur = mean(IBeta(ISurroundingRegion, :));
k = (Vsur - Vdom)/(Betasur - Betadom);

% Estimate diffuse and specular components
Idf = I3c - min(k) * IBeta;
Isp = I3c - Idf;


Idf = uint8(reshape(Idf, height, width, 3));
%figure; imshow(uint8(reshape(I3c, height, width, 3))); title('original'); 
%figure; imshow(uint8(reshape(Idf, height, width, 3))); title('diffuse component');
%figure; imshow(uint8(reshape(Isp, height, width, 3))); title('specular component');


%imwrite(uint8(reshape(Idf, height, width, 3)), 'comp_df.bmp', 'bmp');
%imwrite(uint8(reshape(Isp, height, width, 3)), 'comp_sp.bmp', 'bmp');
toc;
end

