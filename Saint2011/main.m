clc
clear 
close all
Image_dir = './EndoSTTN';
listing = cat(1, dir(fullfile(Image_dir, '*.*g')));
% The final output will be saved in this directory:
result_dir = 'result';
% Preparations for saving results.
if ~exist(result_dir, 'dir'), mkdir(result_dir); end

parfor i_img = 1:length(listing)
    output_filename = fullfile(result_dir, listing(i_img).name);

    % Check if the output file already exists
    if exist(output_filename, 'file')
        fprintf('Skipping %s (already exists in result folder)\n', listing(i_img).name);
        continue;
    end

    img = imread(fullfile(Image_dir,listing(i_img).name));
    specular_mask = SpecularDetectionCharlesAuguste2007(img);
    inpaited_img = InpaintingCharlesAuguste2007(img, specular_mask, 0.05);

    imwrite(inpaited_img, fullfile(result_dir,listing(i_img).name));
end