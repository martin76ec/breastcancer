function [error] = validationError(y, yhat)
    sub = y - yhat;
    sub(sub ~= 0) = 1;
    error = sum(sub) / size(sub, 1);
end
