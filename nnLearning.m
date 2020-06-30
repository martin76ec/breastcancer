function r = nnLearning(X,y,n_classes,hidden_layer_size,lambda,maxIter)
%(2017) by Holger Ortega, based on the script
%by Andrew Ng, from the Machine Learning course on Coursera
%**************************************************************************
%TRAINS A NEURAL NETWORK WITH ONE HIDDEN LAYER
%r - returns a cell array {W1 W2} of parameters after training
%where:
%W1 [#neurons2 x (#neuronsIN + 1)]
%W2 [#neuronsOUT x (#neurons2 + 1)]
%are the weight matrices
%-------------------------------------------------------------------------
%X - Matrix of m training vectors [m x (dimension of feature vectors)]
%y - Vector of known labels [m x 1]
%hidden_layer_size - #neurons in the hidden layer
%n_classes - #classes
%lambda - coeficient for the regularisation
%maxIter - maximum number of iterations that the optimiser function will
%perform

input_layer_size = size(X,2);

%Initialisation ----------------------------------------------------------
initial_W1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_W2 = randInitializeWeights(hidden_layer_size, n_classes);
initial_nn_params = [initial_W1(:) ; initial_W2(:)]; %organise parameters
%as an only vector

%Training ----------------------------------------------------------------

options = optimset('MaxIter', maxIter);
% Create "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   n_classes, X, y, lambda);

% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
[nn_params,~] = fmincg(costFunction, initial_nn_params, options);

% Obtain W1 and W2 back from nn_params
W1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

W2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 n_classes, (hidden_layer_size + 1));

r = {W1,W2};



%==========================================================================
function W = randInitializeWeights(L_in, L_out)
%RANDINITIALIZEWEIGHTS Randomly initialize the weights of a layer with L_in
%incoming connections and L_out outgoing connections
%   W = RANDINITIALIZEWEIGHTS(L_in, L_out) randomly initializes the weights 
%   of a layer with L_in incoming connections and L_out outgoing 
%   connections. 
%
%   Note that W should be set to a matrix of size(L_out, 1 + L_in) as
%   the column row of W handles the "bias" terms
%
% Note: The first row of W corresponds to the parameters for the bias units
%**************************************************by Holger Ortega
%epsilon_init = 0.12;
%W = rand(L_out,1+L_in)*2*epsilon_init - epsilon_init;
W = 0.03*ones(L_out,1+L_in);%NOTE: Actually,this is NOT random initialisation!
% I have fixed the initial W matrix in order to make the exercise more
% predictable, so that all the students will reach the same answers.
% =========================================================================



% =========================================================================
function [J,grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%*********************************by Holger Ortega, based on the layout by
%Andrew Ng, Machine Learning course on Coursera
%**************************************************************************
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
W1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

W2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);

%**********************************************by Holger Ortega

%STEP 1. Feedforward pass
%Append a column of 1s to X
A1 = [ones(m, 1) X];
%Calculate Z2[m x (neurons 2nd layer without the bias)]. Each row is the vector z(2) for each example
Z2 = A1*W1';
%Calculate A2. Activations of layer 2
A2 = sigmoid(Z2);
%Append a column of 1s to A2
A2 = [ones(m, 1) A2];
%Calculate Z3[m x (neurons output)]. Each row is the vector z(3) for each example
Z3 = A2*W2';
%Calculate A3. Activations of layer 3. Each row is the vector a(3) for each
%example. size(A3)=[m x (neurons output)]
A3 = sigmoid(Z3);
%COMPUTE COST
%Process vector y into matrix Y - each value should be transformed into a row of 0s and 1
%Y [m x (neurons output)]
Y = zeros(m,num_labels);
for i=1:m
    if y(i)==0
        Y(i,10)=1;
    else
        Y(i,y(i))=1;
    end
end
%Compute the matrix of terms of the sum in the cost function
termJ = -Y.*log(A3)-(1-Y).*log(1-A3);
%Compute the cost without regularisation
J = sum(sum(termJ))/m;
%Extract first column in each matrix of Theta
W1_noBias=W1(:,2:end);
W2_noBias=W2(:,2:end);
%Compute the regularisation term
regTerm = sum(sum(W1_noBias.^2))+sum(sum(W2_noBias.^2));
regTerm = regTerm*lambda/(2*m);
%Compute the cost with regularisation
J = J + regTerm;

%STEP 2. Errors in layer 3
%Will be stored in delta_3[m x (number output neurons)]
%Each row is the vector delta(3) for each example
delta_3 = A3 - Y;

%STEP 3. Errors in layer 2
%Will be stored in delta_2[m x (number neurons 2nd layer without bias)]
%Each row is the vector delta(2) for each example
g_prime = A2.*(1-A2);
delta_2 = (delta_3*W2).*g_prime;

%STEP 4. Acumulated gradients
%will be stored in:
%Delta2 for the parameters from layer 2 to 3, [#neurons3 x (#neurons2+1)]
%Delta1 for the parameters from layer 1 to 2, [#neurons2 x (#neurons1+1)]
delta_2 = delta_2(:,2:end);
Delta1 = delta_2'*A1;
Delta2 = delta_3'*A2;

%STEP 5. Unregularised gradient
%Theta1_grad [#neurons2 x (#neurons1+1)] and Theta2_grad [#neurons3 x (#neurons2+1)]
Theta1_grad = Delta1/m;
Theta2_grad = Delta2/m;

%STEP 6. Regularised gradient
%Compute the matrices of terms to add
regMatrix1 = lambda*[zeros(size(W1_noBias,1),1),W1_noBias]/m;
regMatrix2 = lambda*[zeros(size(W2_noBias,1),1),W2_noBias]/m;
%Add the regularisation terms
Theta1_grad = Theta1_grad + regMatrix1;
Theta2_grad = Theta2_grad + regMatrix2;

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

%==========================================================================

function g = sigmoid(z)
%SIGMOID Compute sigmoid functoon
%   J = SIGMOID(z) computes the sigmoid of z.
g = 1.0 ./ (1.0 + exp(-z));


%==========================================================================
function [X, fX, i] = fmincg(f, X, options, P1, P2, P3, P4, P5)
% Minimize a continuous differentialble multivariate function. Starting point
% is given by "X" (D by 1), and the function named in the string "f", must
% return a function value and a vector of partial derivatives. The Polack-
% Ribiere flavour of conjugate gradients is used to compute search directions,
% and a line search using quadratic and cubic polynomial approximations and the
% Wolfe-Powell stopping criteria is used together with the slope ratio method
% for guessing initial step sizes. Additionally a bunch of checks are made to
% make sure that exploration is taking place and that extrapolation will not
% be unboundedly large. The "length" gives the length of the run: if it is
% positive, it gives the maximum number of line searches, if negative its
% absolute gives the maximum allowed number of function evaluations. You can
% (optionally) give "length" a second component, which will indicate the
% reduction in function value to be expected in the first line-search (defaults
% to 1.0). The function returns when either its length is up, or if no further
% progress can be made (ie, we are at a minimum, or so close that due to
% numerical problems, we cannot get any closer). If the function terminates
% within a few iterations, it could be an indication that the function value
% and derivatives are not consistent (ie, there may be a bug in the
% implementation of your "f" function). The function returns the found
% solution "X", a vector of function values "fX" indicating the progress made
% and "i" the number of iterations (line searches or function evaluations,
% depending on the sign of "length") used.
%
% Usage: [X, fX, i] = fmincg(f, X, options, P1, P2, P3, P4, P5)
%
% See also: checkgrad 
%
% Copyright (C) 2001 and 2002 by Carl Edward Rasmussen. Date 2002-02-13
%
%
% (C) Copyright 1999, 2000 & 2001, Carl Edward Rasmussen
% 
% Permission is granted for anyone to copy, use, or modify these
% programs and accompanying documents for purposes of research or
% education, provided this copyright notice is retained, and note is
% made of any changes that have been made.
% 
% These programs and documents are distributed without any warranty,
% express or implied.  As the programs were written for research
% purposes only, they have not been tested to the degree that would be
% advisable in any important application.  All use of these programs is
% entirely at the user's own risk.
%
% [ml-class] Changes Made:
% 1) Function name and argument specifications
% 2) Output display
%

% Read options
if exist('options', 'var') && ~isempty(options) && isfield(options, 'MaxIter')
    length = options.MaxIter;
else
    length = 100;
end


RHO = 0.01;                            % a bunch of constants for line searches
SIG = 0.5;       % RHO and SIG are the constants in the Wolfe-Powell conditions
INT = 0.1;    % don't reevaluate within 0.1 of the limit of the current bracket
EXT = 3.0;                    % extrapolate maximum 3 times the current bracket
MAX = 20;                         % max 20 function evaluations per line search
RATIO = 100;                                      % maximum allowed slope ratio

argstr = ['feval(f, X'];                      % compose string used to call function
for i = 1:(nargin - 3)
  argstr = [argstr, ',P', int2str(i)];
end
argstr = [argstr, ')'];

if max(size(length)) == 2, red=length(2); length=length(1); else red=1; end
S=['Iteration '];

i = 0;                                            % zero the run length counter
ls_failed = 0;                             % no previous line search has failed
fX = [];
[f1 df1] = eval(argstr);                      % get function value and gradient
i = i + (length<0);                                            % count epochs?!
s = -df1;                                        % search direction is steepest
d1 = -s'*s;                                                 % this is the slope
z1 = red/(1-d1);                                  % initial step is red/(|s|+1)

while i < abs(length)                                      % while not finished
  i = i + (length>0);                                      % count iterations?!

  X0 = X; f0 = f1; df0 = df1;                   % make a copy of current values
  X = X + z1*s;                                             % begin line search
  [f2 df2] = eval(argstr);
  i = i + (length<0);                                          % count epochs?!
  d2 = df2'*s;
  f3 = f1; d3 = d1; z3 = -z1;             % initialize point 3 equal to point 1
  if length>0, M = MAX; else M = min(MAX, -length-i); end
  success = 0; limit = -1;                     % initialize quanteties
  while 1
    while ((f2 > f1+z1*RHO*d1) | (d2 > -SIG*d1)) && (M > 0) 
      limit = z1;                                         % tighten the bracket
      if f2 > f1
        z2 = z3 - (0.5*d3*z3*z3)/(d3*z3+f2-f3);                 % quadratic fit
      else
        A = 6*(f2-f3)/z3+3*(d2+d3);                                 % cubic fit
        B = 3*(f3-f2)-z3*(d3+2*d2);
        z2 = (sqrt(B*B-A*d2*z3*z3)-B)/A;       % numerical error possible - ok!
      end
      if isnan(z2) | isinf(z2)
        z2 = z3/2;                  % if we had a numerical problem then bisect
      end
      z2 = max(min(z2, INT*z3),(1-INT)*z3);  % don't accept too close to limits
      z1 = z1 + z2;                                           % update the step
      X = X + z2*s;
      [f2 df2] = eval(argstr);
      M = M - 1; i = i + (length<0);                           % count epochs?!
      d2 = df2'*s;
      z3 = z3-z2;                    % z3 is now relative to the location of z2
    end
    if f2 > f1+z1*RHO*d1 | d2 > -SIG*d1
      break;                                                % this is a failure
    elseif d2 > SIG*d1
      success = 1; break;                                             % success
    elseif M == 0
      break;                                                          % failure
    end
    A = 6*(f2-f3)/z3+3*(d2+d3);                      % make cubic extrapolation
    B = 3*(f3-f2)-z3*(d3+2*d2);
    z2 = -d2*z3*z3/(B+sqrt(B*B-A*d2*z3*z3));        % num. error possible - ok!
    if ~isreal(z2) || isnan(z2) || isinf(z2) || z2 < 0   % num prob or wrong sign?
      if limit < -0.5                               % if we have no upper limit
        z2 = z1 * (EXT-1);                 % the extrapolate the maximum amount
      else
        z2 = (limit-z1)/2;                                   % otherwise bisect
      end
    elseif (limit > -0.5) && (z2+z1 > limit)          % extraplation beyond max?
      z2 = (limit-z1)/2;                                               % bisect
    elseif (limit < -0.5) && (z2+z1 > z1*EXT)       % extrapolation beyond limit
      z2 = z1*(EXT-1.0);                           % set to extrapolation limit
    elseif z2 < -z3*INT
      z2 = -z3*INT;
    elseif (limit > -0.5) && (z2 < (limit-z1)*(1.0-INT))   % too close to limit?
      z2 = (limit-z1)*(1.0-INT);
    end
    f3 = f2; d3 = d2; z3 = -z2;                  % set point 3 equal to point 2
    z1 = z1 + z2; X = X + z2*s;                      % update current estimates
    [f2 df2] = eval(argstr);
    M = M - 1; i = i + (length<0);                             % count epochs?!
    d2 = df2'*s;
  end                                                      % end of line search

  if success                                         % if line search succeeded
    f1 = f2; fX = [fX' f1]';
    fprintf('%s %4i | Cost: %4.6e\r', S, i, f1);
    s = (df2'*df2-df1'*df2)/(df1'*df1)*s - df2;      % Polack-Ribiere direction
    tmp = df1; df1 = df2; df2 = tmp;                         % swap derivatives
    d2 = df1'*s;
    if d2 > 0                                      % new slope must be negative
      s = -df1;                              % otherwise use steepest direction
      d2 = -s'*s;    
    end
    z1 = z1 * min(RATIO, d1/(d2-realmin));          % slope ratio but max RATIO
    d1 = d2;
    ls_failed = 0;                              % this line search did not fail
  else
    X = X0; f1 = f0; df1 = df0;  % restore point from before failed line search
    if ls_failed | i > abs(length)          % line search failed twice in a row
      break;                             % or we ran out of time, so we give up
    end
    tmp = df1; df1 = df2; df2 = tmp;                         % swap derivatives
    s = -df1;                                                    % try steepest
    d1 = -s'*s;
    z1 = 1/(1-d1);                     
    ls_failed = 1;                                    % this line search failed
  end
  if exist('OCTAVE_VERSION')
    fflush(stdout);
  end
end
fprintf('\n');