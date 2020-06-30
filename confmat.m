function [cmat] = confmat(y, yhat)
    class = unique(y);
    class_dim = size(unique(y), 1);
    cmat = zeros(class_dim);
    for i = 1:class_dim
        real = class(i);
        [r, c] = find(y == real);
        for j = 1:class_dim
            predicted = class(j);
            for k = 1:size(r, 1)
                position = r(k);
                equal = predicted == yhat(r(k));
                cmat(i, j) = cmat(i, j) + equal;
            end
        end
    end
end
