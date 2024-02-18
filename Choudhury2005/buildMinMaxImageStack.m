function [minStack,maxStack] = buildMinMaxImageStack(gradientMagnitude,maxLevel)

minStack = zeros(...
    size(gradientMagnitude,1),...
    size(gradientMagnitude,2),...
    maxLevel);
maxStack = minStack;
minImg1 = minStack(:,:,1);
maxImg1 = maxStack(:,:,1);

for row = 1:size(minStack,1)
    for col = 1:size(minStack,2)
        outMin = 1E12;
        outMax = -1E12;
        for n = max(row-1,1):min(row+2,size(minStack,1))
            for m = max(col-1,1):min(col+2,size(minStack,2))
                outMin = min(gradientMagnitude(n,m),outMin);
                outMax = max(gradientMagnitude(n,m),outMax);
            end
        end
        minImg1(row,col) = outMin;
        maxImg1(row,col) = outMax;
    end
end

for ii = 2:size(minStack,3)
    minImg1 = minStack(:,:,ii-1);
    maxImg1 = maxStack(:,:,ii-1);
    
    minImg2 = minStack(:,:,ii);
    maxImg2 = maxStack(:,:,ii);
    
    for row = 1:size(minStack,1)
        for col = 1:size(minStack,2)
            outMin = 1E12;
            outMax = -1E12;
            for n = max(row-1,1):min(row+2,size(minStack,1))
                for m = max(col-1,1):min(col+2,size(minStack,2))
                    outMin = min(minImg1(n,m),outMin);
                    outMax = max(maxImg1(n,m),outMax);
                end
            end
            minImg2(row,col) = outMin;
            maxImg2(row,col) = outMax;
        end
    end
end

end