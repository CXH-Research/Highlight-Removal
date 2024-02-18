function [xGradient,yGradient] = computeGradients(inputImage)
xKernel = [-1,1];
yKernel = [-1;1];
xGradient = filter2(xKernel,inputImage,'same');
yGradient = filter2(yKernel,inputImage,'same');
end