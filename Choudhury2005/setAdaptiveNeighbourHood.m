function adaptiveNeighbourhood = setAdaptiveNeighbourHood(...
    gradientMagnitude,sigmaR,maxLevel)

adaptiveNeighbourhood = zeros(size(gradientMagnitude));
[minStack,maxStack] = buildMinMaxImageStack(gradientMagnitude,maxLevel);

for row = 1:size(gradientMagnitude,1)
    for col = 1:size(gradientMagnitude,2)
        upperThreshold = gradientMagnitude(row,col) + sigmaR;
        lowerThreshold = gradientMagnitude(row,col) - sigmaR;
        for lev = 1:maxLevel
            minImg = minStack(:,:,lev);
            maxImg = maxStack(:,:,lev);
            if maxImg(row,col)>upperThreshold || minImg(row,col)<lowerThreshold
                break;
            end
        end
        adaptiveNeighbourhood(row,col) = 2^(lev-1);
    end
end
end