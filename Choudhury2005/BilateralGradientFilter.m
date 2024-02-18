function [xGradientSmooth,yGradientSmooth] = BilateralGradientFilter(...
    xGradient,yGradient,gradientMagnitude,sigmaC,sigmaR,epsilon)

xGradientSmooth = zeros(size(xGradient));
yGradientSmooth = xGradientSmooth;
domainConst = -2*sigmaC*sigmaC;
rangeConst = -2*sigmaR*sigmaR;
halfSize = ceil(sigmaC/2);
domainWeight = zeros(halfSize,halfSize);

for row = 1:size(domainWeight,1)
    for col = 1:size(domainWeight,2)
        diff_ = col*col+row*row;
        domainWeight(row,col) = exp(diff_/domainConst);
    end
end

for row = 1:size(gradientMagnitude,1)
    for col = 1:size(gradientMagnitude,2)
        normFactor = 0;
        tmpX = 0;
        tmpY = 0;
        g2 = gradientMagnitude(row,col);
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
                    if localX >= size(gradientMagnitude,2)+1
                        continue
                    end
                    localY = row +n;
                    if localY < 1
                        continue
                    end
                    if localY >= size(gradientMagnitude,1)+1
                        continue
                    end
                    g1 = gradientMagnitude(localY,localX);
                    gradDiffSq = (g1-g2)^2;
                    rangeWeight = exp(gradDiffSq/rangeConst);
                    if rangeWeight < epsilon
                        continue
                    end
                    tmpX = tmpX + xGradient(localY,localX)*dWeight*rangeWeight;
                    tmpY = tmpY + yGradient(localY,localX)*dWeight*rangeWeight;
                    normFactor = normFactor + dWeight*rangeWeight;
                end
            end
        end
        xGradientSmooth(row,col) = tmpX/normFactor;
        yGradientSmooth(row,col) = tmpY/normFactor;
    end
end

end