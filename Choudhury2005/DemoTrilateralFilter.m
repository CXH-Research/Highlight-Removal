load('durer');
inputImage = imresize(X,0.25);

tic
sigmaC = 8;
epsilon = 0.1;
outputImage = trilateralFilter(inputImage,sigmaC,epsilon);
toc

figure(1);
subplot(1,2,1); imagesc(outputImage); title('Filtered'); colormap(gray); axis equal
subplot(1,2,2); imagesc(inputImage); title('Original'); colormap(gray); axis equal