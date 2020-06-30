clear;
close all;
clc;


dataC = csvread('breast-cancer-wisconsin.data');

#Data extraction
X=[dataC(:, 2), dataC(:, 3), dataC(:, 4), dataC(:, 5), dataC(:, 2), dataC(:, 6), dataC(:, 8), dataC(:, 9), dataC(:, 10)];
y =[dataC(:, 11)];
y(y == 2) = 1;
y(y == 4) = 2;

# Splitting data
X_train = X(1:489, :);
y_train = y(1:489, :);
X_val = X(490:594, :);
y_val = y(490:594, :);
X_test = X(595:end, :);
y_test = y(595:end, :);

#Validation
hidden_layer_size = 1:5;
lambda_values = [0.5, 1, 1.5];
iterations = 1:20;
validation_matrix = zeros(300, 4);
count = 1;
for i = 1:5
    for j = 0.5:0.5:1.5
        for k = 1:20
            weights = nnLearning(X_train, y_train, 2, i, j, k);
            yhat = nnOutput(X_val, weights(1, 1){1, :}, weights(1, 2){1, :});
            error = validationError(y_val, yhat);
            validation_matrix(count, :) = [i, j, k, error];
            count = count + 1;
        end
    end
end

#Get optimal values
min_error = min(validation_matrix(:, 4));
[r, c] = find(validation_matrix(:, 4) == min_error);
optimal_hiperparameters = validation_matrix(r, :)(1, :);

#Recover optimal weights
weights = nnLearning(X_train, y_train, 2, optimal_hiperparameters(1), optimal_hiperparameters(2), optimal_hiperparameters(3));
W1_opt = weights(1, 1){1, :};
W2_opt = weights(1, 2){1, :};

#Testing
yhat_test = nnOutput(X_test, W1_opt, W2_opt);
error = validationError(y_test, yhat_test);
cmat = confmat(y_test, yhat_test);





