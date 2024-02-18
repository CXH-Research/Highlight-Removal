function outputImage = DetailBilateralFilter(inputImage,adaptiveRegion,...
    xGradientSmooth,yGradientSmooth,sigmaC,sigmaR,maxDomainSize,epsilon)

outputImage = zeros(size(inputImage));
domainConst = -2*sigmaC*sigmaC;
rangeConst = -2*sigmaR*sigmaR;

domainWeight = zeros(maxDomainSize,maxDomainSize);

for row = 1:size(domainWeight,1)
    for col = 1:size(domainWeight,2)
        diff_ = row*row+col*col;
        domainWeight(row,col ) = exp(diff_/domainConst);
    end
end

for row = 1:size(inputImage,1)
    for col = 1:size(inputImage,2)
        normFactor = 0;
        tmp = 0;
        halfSize = min(adaptiveRegion(row,col),maxDomainSize);
        coeffA = xGradientSmooth(row,col);
        coeffB = yGradientSmooth(row,col);
        coeffC = inputImage(row,col);
        for n = -halfSize:halfSize
            for m = -halfSize:halfSize
                if (n && m) 
                    dWeight = domainWeight(abs(n),abs(m));
                    if dWeight < epsilon
                        continue
                    end
                    localX = col + m;
                    if localX < 1
                        continue
                    end
                    if localX >= size(inputImage,2)+1
                        continue
                    end
                    localY = row +n;
                    if localY < 1
                        continue
                    end
                    if localY >= size(inputImage,1)+1
                        continue
                    end
                    detail = inputImage(localY,localX) - coeffA*m - ...
                        coeffB*n - coeffC;
                    rangeWeight = exp(detail^2 / rangeConst);
                    tmp = tmp+detail*dWeight*rangeWeight;
                    normFactor = normFactor + dWeight*rangeWeight;
                end
            end
        end
        outputImage(row,col) = tmp/normFactor + coeffC;
    end
end

end