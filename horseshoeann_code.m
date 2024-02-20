load normalisedhorseshoe.csv                                                % Loads the normalized horseshoe data
normData = normalize(normalisedhorseshoe, 'range', [0.1, 0.9]);             % Scales the horseshoe data samples into a range from 0.1 - 0.9
hspInp = normalisedhorseshoe(:, 1:4);                                       % Sets input variables to columns 1 to 4
hsOut = normalisedhorseshoe(:, 5);                                          % Sets output variable (target) to column 5
x = hspInp';                                                                % Assigns input value to x
t = hsOut';                                                                 % Assigns output (target) value to t
 
% Defines different Neurons, Learning rates & Epoch Values (Hyperparameters)
neurons = [15, 35, 50];                                                     % Array containing 3 values for Neurons 
learningrate = [0.01, 0.05, 0.1];                                           % Array containing 3 values for Learning rate
epochs = [5, 15, 30];                                                       % Array containing 3 values for Epoch number

bestMSE = inf;                                                              % Sets it to infinity, making sure MSE will always be lowest value obtained

% Sets the performance array as a structure array
perfArray = struct('neurons', [], 'learningrate', [], 'epochs', []);        % Creates an empty structure with empty arrays for the hyperparamaters to be stored

% Nested For Loops
for n = 1:numel(neurons)                                                    % Nested for loops, will go through each combination one by one
    for lr = 1:numel(learningrate)                                          % numel shows the number of elements in an array
        for ep = 1:numel(epochs)                                            % Generates range of values
                                                      
            % Creating the Neural Network
             net = feedforwardnet(neurons(n));                              % Creates a feedforward neural network with amount of neurons in hidden layer
             net = init(net);                                               % Initialises the hyperparameters (wieghts & biases) before training
           

% Sets the Transfer Functions for the Hidden Layer & Output layer
            net.layers{1}.transferFcn = 'poslin';                           % hiddenlayer 1    
            net.layers{end}.transferFcn = 'purelin';                        % Final layer

            TF = isoutlier(normData);                                       % outlines which data samples are outliers
            XO = filloutliers(normData,'linear', 'mean') ;                  % Averages the outlier data using linear interpolation method from other horseshoe data (mean)
                 
            net.divideFcn = 'dividerand';                                   % Function used to divide data samples Randomly
            net.divideParam.trainRatio = 70/100;                            % training dataset ratio (70%)
            net.divideParam.valRatio = 15/100;                              % validation dataset (15%)
            net.divideParam.testRatio = 15/100;                             % testing dataset (15%)

            net.trainParam.epochs = epochs(ep);                             % Set the number of iterations 
            net.trainParam.lr = learningrate(lr);                           % Sets the learning rate

            net.performFcn = 'mse';                                         % Performance function                                  
            net.trainFcn = 'trainlm';                                       % Training function


            % Trains the network
            [net, tr] = train(net, x, t);                                   % Trains net using hspInp' and hsOut'

            % Tests the network
            simT = sim(net, x);                                             % makes predictions (simulations) on the hspInp' data

            % Calculates the MSE Value
            perf = mse(net, t, simT);                                       % Compares predicted values(SimT) with Target Values (hsOut')

            if perf < bestMSE                                               % If current combination MSE is lover than bestMSE
                bestMSE = perf;                                             % Updates bestMSE if a new good (lower) MSE value is obtained

                % Updates the performance arrays
                perfArray(n, lr, ep).neurons = neurons(n);                  % Updates Neuron value from the n Array from the current combination ran 
                perfArray(n, lr, ep).learningrate = learningrate(lr);       % Updates learning Rate value from the lr Array from the current combination ran
                perfArray(n, lr, ep).epochs = epochs(ep);                   % Updates Epochs value from the ep Array from the current combination ran
                perfArray(n, lr, ep).MSE = perf;                            % Updates the MSE field from the perfArray with the MSE value recently obtained from current combination ran
            end
        end
    end
end



bestCombination = [];                                                       % sets variable to an empty array. Stores best hyperparameter combination.
for element = 1:numel(perfArray)                                            % element is a loop counter, will run until all 27 combinations are done
    if perfArray(element).MSE == bestMSE                                    % Will set the current combination trained MSE value as the bestMSE 
        bestCombination = perfArray(element);                               % If current MSE Value is lower than bestMSE, then this code will update bestCombination value (as we are trying to find lowest MSE value)
    end
end
           
% Show the Best combination of the 3 hyperparameters we have acheived 
fprintf('Best Neuron Number: %d\n', bestCombination.neurons);               % Prints the value for best Hidden Layer. (%d) specifies for an Integer
fprintf('Best Learning Rate Value: %.2f\n', bestCombination.learningrate);  % Prints the value for best Learning Rate. (%.2f) specifies 2 decimal spots to be returned in answer
fprintf('Best Epochs Number: %d\n', bestCombination.epochs);                % Prints the value for best Epoch Number. (%d) specifies for an Integer
fprintf('Lowest MSE Value Obtained: %.4f\n', bestCombination.MSE);          % Prints the value for best (lowest) MSE Value obtained (%.4f) specifies 4 decimal spots to be returned in answer


% Displays the original and altered(no outliers) horseshoe data
figure;                                                                     % creates a new window to show the graphs                                                                   
subplot(2,1,1)                                                              % Creats a grid with 2 rows and 1 column (2x1 grid). First scatter plot is in row 1
plot(x, t, 'o');                                                            % creates a scatter plot (circle)
title('Original Horseshoe Data (with Outliers)');                           % Sets the name of scatter plot 1

subplot(2,1,2)                                                              % Creats a grid with 2 rows and 1 column (2x1 grid). Second scatter plot is in row 2
plot(XO, 'o');                                                              % Creates a scatter plot of X (circle)
title('Horseshoe Data (without Outliers)');                                 % Sets the name of scatter plot 2



