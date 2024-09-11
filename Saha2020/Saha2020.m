clear all;close all;clc;
clc
clear 
close all
Image_dir = '../EndoSTTN';
listing = cat(1, dir(fullfile(Image_dir, '*.*g')));
% The final output will be saved in this directory:
result_dir = 'result';
% Preparations for saving results.
if ~exist(result_dir, 'dir'), mkdir(result_dir); end

for i_img = 1:length(listing)
    Input = imread(fullfile(Image_dir,listing(i_img).name));
    sfi = fix(Input);
    imwrite(sfi, fullfile(result_dir,listing(i_img).name));
end


function [res] = fix(img)
    [w,h,d] = size(img);
    
    for i = 1 : 1 : w
        for j = 1 : 1 : h
            sum(i,j) = min(min(img(i,j,:)));
            
        end
    end
     offset = mean2(sum);
     ini_offset = offset;
    for i = 1 : 1 : w
        for j = 1 : 1 : h
            if(max(max(img(i,j,:))) - min(min(img(i,j,:))) + ini_offset >255)
                n_offset = ini_offsett - ((max(max(img(i,j,:))) - min(min(img(i,j,:))) + offset) - 255);
                MSF(i,j,:) = img(i,j,:) - min(min(img(i,j,:))) + n_offset;
            end
            if(ini_offset > max(max(img(i,j,:))))
                n_offset =  max(max(img(i,j,:)));
                MSF(i,j,:) = img(i,j,:) - min(min(img(i,j,:))) + n_offset;
            else
                MSF(i,j,:) = img(i,j,:) - min(min(img(i,j,:))) + ini_offset;
                
            end
                
            
        end
    end
    res = uint8(MSF);
% I2(i,j,:) = I1(i,j,:) - min(min(I1(i,j,:))) + offset;
end
