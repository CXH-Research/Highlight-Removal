clc
clear 
close all
Image_dir = '../EndoSRR';
listing = cat(1, dir(fullfile(Image_dir, '*.*g')));
% The final output will be saved in this directory:
result_dir = 'result';
% Preparations for saving results.
if ~exist(result_dir, 'dir'), mkdir(result_dir); end

for i_img = 1:length(listing)
    sfi = fix(im2double(imread(fullfile(Image_dir,listing(i_img).name))), @Shen2013);
    imwrite(sfi, fullfile(result_dir,listing(i_img).name));
end



function I_d = Shen2013(I)
%Shen2013 I_d = Shen2013(I)
%  You can optionally edit the code to use kmeans instead of the clustering
%  function proposed by the author.
%  
%  This method should have equivalent functionality as
%  `sp_removal.cpp` distributed by the author.
%  
%  See also SIHR, Shen2008, Shen2009.

assert(isa(I, 'float'), 'SIHR:I:notTypeSingleNorDouble', ...
    'Input I is not type single nor double.')
assert(min(I(:)) >= 0 && max(I(:)) <= 1, 'SIHR:I:notWithinRange', ...
    'Input I is not within [0, 1] range.')
[n_row, n_col, n_ch] = size(I);
assert(n_row > 1 && n_col > 1, 'SIHR:I:singletonDimension', ...
    'Input I has a singleton dimension.')
assert(n_ch == 3, 'SIHR:I:notRGB', ...
    'Input I is not a RGB image.')

height = size(I, 1);
width = size(I, 2);
I = reshape(I, [height * width, 3]);

Imin = min(I, [], 2);
Imax = max(I, [], 2);
Iran = Imax - Imin;

umin_val = mean2(Imin);

Imask = Imin > umin_val;

Ich_pseudo = zeros([height * width, 2]);
frgb = zeros([height * width, 3]);
crgb = frgb;
srgb = zeros([height * width, 1]);

frgb(Imask, :) = I(Imask, :) - Imin(Imask) + umin_val;
srgb(Imask) = sum(frgb(Imask, :), 2);
crgb(Imask, :) = frgb(Imask, :) ./ srgb(Imask);

Ich_pseudo(Imask, 1) = min(min(crgb(Imask, 1), crgb(Imask, 2)), crgb(Imask, 3));
Ich_pseudo(Imask, 2) = max(max(crgb(Imask, 1), crgb(Imask, 2)), crgb(Imask, 3));

% num_clust = 3;
% Iclust = zeros([height * width, 1]);
% Iclust(Imask) = kmeans([Ich_pseudo(Imask, 1), Ich_pseudo(Imask, 2)], num_clust, 'Distance', 'cityblock', 'Replicates', ceil(sqrt(num_clust)));
th_chroma = 0.3;
[Iclust, num_clust] = pixel_clustering(Ich_pseudo, Imask, width, height, th_chroma);

ratio = zeros([height * width, 1]);
Iratio = zeros([height * width, 1]);

N = width * height;
EPS = 1e-10;
th_percent = 0.5;

for k = 1:num_clust
    num = 0;
    for i = 1:N
        if (Iclust(i) == k && Iran(i) > umin_val)
            ratio(num+1) = Imax(i) / (Iran(i) + EPS);
            num = num + 1;
        end
    end

    if num == 0
        continue
    end

    tmp = sort(ratio(1:num));
    ratio_est = tmp(round(num*th_percent)+1);

    for i = 1:N
        if (Iclust(i) == k)
            Iratio(i) = ratio_est;
        end
    end
end

I_s = zeros([height * width, 1]);
I_d = I;

for i = 1:N
    if (Imask(i) == 1)
        uvalue = (Imax(i) - Iratio(i) * Iran(i)); % round( . )
        I_s(i) = max(uvalue, 0);
        fvalue = I(i, 1) - I_s(i);
        I_d(i, 1) = (clip(fvalue, 0, 1)); % round
        fvalue = I(i, 2) - I_s(i);
        I_d(i, 2) = (clip(fvalue, 0, 1)); % round
        fvalue = I(i, 3) - I_s(i);
        I_d(i, 3) = (clip(fvalue, 0, 1)); % round
    end
end

% I_s = reshape(I_s, [height, width]);
I_d = reshape(I_d, [height, width, 3]);

end


function [Iclust, num_clust] = pixel_clustering(Ich_pseudo, Imask, width, height, th_chroma)
MAX_NUM_CLUST = 100;

label = 0;
c = zeros([2, 1]);

clust_mean = zeros([MAX_NUM_CLUST, 2]);
num_pixel = zeros([MAX_NUM_CLUST, 1]);

N = width * height;

Idone = zeros([height * width, 1], 'logical');
Iclust = zeros([height * width, 1], 'uint8');

for i = 1:N
    if (Idone(i) == 0 && Imask(i) == 1)
        c(1) = Ich_pseudo(i, 1);
        c(2) = Ich_pseudo(i, 2);
        label = label + 1;
        for j = i:N
            if (Idone(j) == 0 && Imask(j) == 1)
                dist = abs(c(1)-Ich_pseudo(j, 1)) + abs(c(2)-Ich_pseudo(j, 2));
                if (dist < th_chroma)
                    Idone(j) = 1;
                    Iclust(j) = label;
                end
            end
        end
    end
end

num_clust = label;

if num_clust > MAX_NUM_CLUST
    return
end

for i = 1:N
    k = Iclust(i);
    if (k >= 1 && k <= num_clust)
        num_pixel(k) = num_pixel(k) + 1;
        clust_mean(k, 1) = clust_mean(k, 1) + Ich_pseudo(i, 1);
        clust_mean(k, 2) = clust_mean(k, 2) + Ich_pseudo(i, 2);
    end
end

for k = 1:num_clust
    clust_mean(k, 1) = clust_mean(k, 1) / num_pixel(k);
    clust_mean(k, 2) = clust_mean(k, 2) / num_pixel(k);
end

for i = 1:N
    if Imask(i) == 1
        c(1) = Ich_pseudo(i, 1);
        c(2) = Ich_pseudo(i, 2);
        dist_min = abs(c(1)-clust_mean(2, 1)) + abs(c(2)-clust_mean(2, 2));
        label = 1;
        for k = 2:num_clust
            dist = abs(c(1)-clust_mean(k, 1)) + abs(c(2)-clust_mean(k, 2));
            if (dist < dist_min)
                dist_min = dist;
                label = k;
            end
        end
        Iclust(i) = label;
    end
end

end

function y = clip(x, lb, ub)
y = min(ub, max(lb, x));
end


function I_d = fix(I, AuthorYEAR)
%fix I_d = fix(I, AuthorYEAR)
%  This function enhances the diff. comp. I_d obtained by AuthorYEAR method.
%
%  AuthorYEAR is optional (defaults to Shen2013), but can also be specified
%  as a valid function handle, e.g. @Yang2010, @Shen2008.
%
%  Example:
%    J = im2double(imread('toys.ppm'));
%    J_d = fix(J, @Yoon2006);
%    imshow(J_d)
%
%  There were no major ambiguities in my interpretation of the corresponding
%  paper.
%
%  There's only a mention of applying SVD to obtain the specular component
%  after calculating a diffuse component by applying an existing method.
%  However, all operations regard $\beta S$, i.e. I_s which is I - I_d under
%  DRM assumption (that they're linearly additive). Hence, we skip this
%  step (Part 2).
%  
%  See also SIHR, Yang2010, Shen2013.

if nargin == 2
    if isempty(AuthorYEAR)
        AuthorYEAR = @Shen2013;
    else
        assert(isa(AuthorYEAR, 'function_handle'), 'SIHR:I:notTypeFunctionHandle', ...
            'Input AuthorYEAR is not of type function_handle.')
        my_f = functions(AuthorYEAR);
        if isempty(my_f.file)
            warning(['Undefined function ''', func2str(AuthorYEAR), '''.', sprintf('\n'), ...
                '         Defaulting to ''Shen2013''.'])
            AuthorYEAR = @Shen2013;
        end
    end
elseif nargin == 1
    AuthorYEAR = @Shen2013;
end

[n_row, n_col, ~] = size(I);

% DRM: I = I_d + I_s
I_d = feval(AuthorYEAR, I);
I_s = my_clip(I-I_d, 0, 1);

I_d_m_1 = I_d;
% I_d_init = I_d; % %DEBUGVAR%

% Table 1
omega = 0.3;
k = 10;
epsilon = 0.2; % RMSE convergence criteria
iter_count = uint8(0);
max_iter_count = uint8(5);

H_low = fspecial('average', 3);
H_h_emph = -k * H_low;
H_h_emph(2, 2) = 1 + k - k * H_low(2, 2);
Theta = my_clip(imfilter(I, H_h_emph, 'symmetric'), 0, 1);

while true
    Upsilon_d = my_clip(imfilter(I_d, H_h_emph, 'symmetric'), 0, 1);
    Upsilon_s = my_clip(imfilter(I_s, H_h_emph, 'symmetric'), 0, 1);
    Upsilon = my_clip(Upsilon_d+Upsilon_s, 0, 1);

    err_diff = sum(double(Upsilon_d > Theta), 3) >= 3; % 1

    counts = imhist(I_s(:, :, 1));
    I_s_bin = im2bw(I_s, otsuthresh(counts)); %#ok
    N_s = nnz(I_s_bin);
    if N_s == 0
        break
    end
    N_s = 2 * ceil(sqrt(N_s)/2) + 1;
    % N_s = 3;
    center = floor(([N_s, N_s] + 1)/2);

    [row, col] = ind2sub(size(err_diff), find(err_diff));

    offset_r = center(1) - 1;
    offset_c = center(2) - 1;

    for ind = 1:nnz(err_diff)
        nh_r = max(1, row(ind)-offset_r):min(row(ind)+offset_r, n_row);
        nh_c = max(1, col(ind)-offset_c):min(col(ind)+offset_c, n_col);

        nh_I = reshape(I(nh_r, nh_c, :), [], 3);
        nh_Theta = reshape(Theta(nh_r, nh_c, :), [], 3);
        nh_Upsilon = reshape(Upsilon(nh_r, nh_c, :), [], 3);

        center_p = reshape(I(row(ind), col(ind), :), [], 3);

        Phi_I = sum((center_p - nh_I).^2, 2);
        Phi_Th_Up = sum((nh_Theta - nh_Upsilon).^2, 2);

        Phi = omega * Phi_I + (1 - omega) * Phi_Th_Up;

        [~, plausible] = min(Phi(:));
        [p_row, p_col] = ind2sub([length(nh_r), length(nh_c)], plausible);

        I_d(row(ind), col(ind), :) = ...
            I_d_m_1(nh_r(p_row), ...
            nh_c(p_col), :);
    end

    iter_count = iter_count + 1;

    I_s = my_clip(I-I_d, 0, 1);

    if sqrt(immse(I_d, I_d_m_1)) < epsilon || ...
            iter_count >= max_iter_count
        break
    end

    I_d_m_1 = I_d;
end

% figure(1), imshow([I_d_init, I_d])
% figure(2), imshow(10*abs(I_d-I_d_init))

end