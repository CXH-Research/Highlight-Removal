function outputImage = trilateralFilter(inputImage,sigmaC,epsilon)
%TRILATERALFILTER Two-dimensional trilateral filter.
%   outputImage = trilateralFilter(inputImage,sigmaC,epsilon)
%   filters the data in inputImage with the 2-D trilateral filter by 
%   P. Choudhury and J. Tumblin, http://www.cs.northwestern.edu/~jet/publications.html
%
%   This implementation is a conversion of the OpenCV implementation by Tobi Vaudrey
%   http://www.cs.auckland.ac.nz/~tvau003/openCV-Examples.html
%
%   Pekka Astola, pekka.astola@tut.fi

beta = 0.15;
[xGradient,yGradient] = computeGradients(inputImage);
gradientMagnitude = computeMagnitude(xGradient,yGradient);
minGrad = min(min(gradientMagnitude));
maxGrad = max(max(gradientMagnitude));
sigmaR = beta*(maxGrad-minGrad);
maxLUT = round(sigmaC*sqrt(abs(log(epsilon))))+1;
maxLevel = ceil(log2(2*maxLUT+1));

adaptiveNeighbourhood = setAdaptiveNeighbourHood(gradientMagnitude,...
    sigmaR,maxLevel);
[xGradientSmooth,yGradientSmooth] = BilateralGradientFilter(...
    xGradient,yGradient,gradientMagnitude,sigmaC,sigmaR,epsilon);
outputImage = DetailBilateralFilter(inputImage,adaptiveNeighbourhood,...
    xGradientSmooth,yGradientSmooth,sigmaC,sigmaR,maxLUT,epsilon);

end