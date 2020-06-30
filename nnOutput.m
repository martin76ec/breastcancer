function [yhat] = nnOutput(X, W1, W2)
    Z1 = [ones(size(X, 1), 1), X];
    S2 = Z1 * W1';
    A2 = zeros(size(S2));
    
    for i = 1:size(S2, 1)
        for j = 1:size(S2, 2)
            A2(i, j) = 1 / (1 + e^-S2(i, j));
        end
    end
    
    Z2 = [ones(size(A2, 1), 1), A2];
    S3 = Z2 * W2';
    A3 = zeros(size(S3));
    
    for i = 1:size(S3, 1)
        for j = 1:size(S3, 2)
            A3(i, j) = 1 / (1 + e^-S3(i, j));
        end
    end
    
    [c, i] = max(A3');
    yhat = i';
    
end
