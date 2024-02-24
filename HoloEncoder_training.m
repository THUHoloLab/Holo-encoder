clearvars  -except dlnet averageGrad averageSqGrad;
close all; clc

addpath('./functions');

%% data
% load network
load HoloEncoder_trained.mat

% load dataset
trainImagePath = 'D:\Datasets\DIV2K_train_HR';
validImagePath = 'D:\Datasets\DIV2K_valid_HR';
trainimds = imageDatastore(trainImagePath,'IncludeSubfolders',true);
trainAugimds = augmentedImageDatastore([2160 3840],trainimds,'ColorPreprocessing',"rgb2gray");
validimds = imageDatastore(validImagePath,'IncludeSubfolders',true);
validAugimds = augmentedImageDatastore([2160 3840],validimds,'ColorPreprocessing',"rgb2gray");

% initialize plot
[ax1,ax2,lineLossTotal,lineLossValid]=initializePlots();
plotFrequency = 10;

%% training parameters
numEpochs = 10;
miniBatchSize = 4;
trainAugimds.MiniBatchSize = miniBatchSize;
validAugimds.MiniBatchSize = 1;
averageGrad = [];
averageSqGrad = [];
numIterations = floor(trainAugimds.NumObservations/miniBatchSize)*numEpochs*10;

learnRate = 0.001;
gradientDecayFactor = 0.9;
squaredGradientDecayFactor = 0.999;
executionEnvironment = "auto";

%% training
iteration = 0;
start = tic;

% Loop over epochs.
for i = 1:numEpochs
    
    % Reset and shuffle datastore.
    reset(trainimds);
    trainimds = shuffle(trainimds);
    
    % Loop over mini-batches.
    while hasdata(trainAugimds)
                
        % Read mini-batch of data.
        data = read(trainAugimds);
                       
        % Ignore last partial mini-batch of epoch.
        if size(data,1) < miniBatchSize
            continue
        end
     
        % Extract the images from data store into a cell array.
        images = data{:,1};
        
        % Concatenate the images along the 4th dimension.
        X = cat(4,images{:});
        X = single(X);
        
        % Convert mini-batch of data to dlarray and specify the dimension labels
        % 'SSCB' (spatial, spatial, channel, batch).
        dlX = dlarray(X, 'SSCB');
        
        % If training on a GPU, then convert data to gpuArray.
        if (executionEnvironment == "auto" && canUseGPU) || executionEnvironment == "gpu"
            dlX = gpuArray(dlX);
        end

        for N = 1:10
        iteration = iteration + 1;
        % Evaluate model gradients.
        [gradients,~,~,loss] = dlfeval(@modelGradients,dlnet,dlX);

        % Update the network parameters using the Adam optimizer.
        [dlnet,averageGrad,averageSqGrad] = ...
            adamupdate(dlnet,gradients,averageGrad,averageSqGrad,iteration,...
            learnRate, gradientDecayFactor, squaredGradientDecayFactor);
        

        addpoints(lineLossTotal,iteration,double(gather(extractdata(loss))))
%         addpoints(lineLossPercept,iteration,double(gather(extractdata(lossPercept))))
%         addpoints(lineLossNpcc,iteration,double(gather(extractdata(lossNpcc))))
        
        % Every plotFequency iterations, plot the training progress.
        if iteration == 1 || mod(iteration,plotFrequency) == 0
            reset(validAugimds);
            validAugimds = shuffle(validAugimds);
            validData = read(validAugimds);
            validImages = validData{:,1};

            VX = single(validImages{1});
            dlVX = dlarray(VX, 'SSCB');
        
            % If training on a GPU, then convert data to gpuArray.
            if (executionEnvironment == "auto" && canUseGPU) || executionEnvironment == "gpu"
                dlVX = gpuArray(dlVX);
            end

            % Use the transformed validation image computed previously.
            [gradients,dlYc,dlVH,lossValid] = dlfeval(@modelGradients,dlnet,dlVX);
            dlYc = rescale(dlYc,0,255);
            dlVH = rescale(dlVH,0,255);
            
            % To use the function imshow, convert to uint8.
            targetImage = imresize(uint8(gather(extractdata(dlVX))),0.25);
            reconImage = imresize(uint8(gather(extractdata(dlYc))),0.5);
            holoImage = imresize(uint8(gather(extractdata(dlVH))),0.25);
            
            % Plot the input image and the output image and increase size
            imshow(imtile({targetImage,reconImage,holoImage},'GridSize', [1 3]),'Parent',ax2);
            addpoints(lineLossValid,iteration,double(gather(extractdata(lossValid))))
        end
        
        % Display time elapsed since start of training and training completion percentage.
        D = duration(0,0,toc(start),'Format','hh:mm:ss');
        completionPercentage = round(iteration/numIterations*100,2);
        title(ax1,"Epoch: " + i + ", Iteration: " + iteration +" of "+ numIterations + "(" + completionPercentage + "%)"+...
            ", LearnRate: "+ learnRate + ", Elapsed: " + string(D))
        drawnow
        
        end
      
    end
    
%     learnRate = learnRate*0.9;
end

save('HoloEncoder_trained.mat','dlnet');

%% Prediction

X = imread('D:\Dataset\DIV2K_valid_HR\0801.png');
X = rgb2gray(X);
X = imresize(X,[2160 3840]);
X = single(X);
dlX = dlarray(X,'SSCB');

[dlZ,dlY] = forward(dlnet,dlX,'Outputs',{'tanh','I'});
Y = extractdata(dlY);
Z = extractdata(dlZ);
figure,imshow(Y,[]);
figure,imshow(Z,[]);

%% netloss

function [gradients,dlYc,dlH,loss] = modelGradients(dlnet,dlX)

    [dlH,dlY] = forward(dlnet,dlX,'Outputs',{'tanh','I'});
    dlYc = dlY(1239:2601,709:3131,:,:);
    X = gather(extractdata(dlX));
    Xc = imresize(X,[1362 2422]);
    Xc = padarray(Xc,[1239 709]);
    dlXc = dlarray(Xc, 'SSCB');
    lossNpcc = npccLoss(dlY,dlXc);

    % Apply weights.
%     lossPercept = weightPercept * lossPercept;
    loss = (lossNpcc + 1)/2;

    % Calculate the total loss.
%     loss = lossPercept + lossNpcc;

    gradients = dlgradient(loss,dlnet.Learnables);

end

function lossContent = perceptualLoss(dlnetLoss,dlY,dlX)

% Extract activations.
dlYActivations = forward(dlnetLoss,dlY,'Outputs','relu3_3');
dlXActivations = forward(dlnetLoss,dlX,'Outputs','relu3_3');

% Calculate the mean square error between activations.
lossContent = mean((dlYActivations - dlXActivations).^2,'all');

end

function loss = npccLoss(dlX,dlY)

X0 = dlX - mean(dlX,[1 2]);
Y0 = dlY - mean(dlY,[1 2]);
X0_norm = sqrt(sum(X0.^2,[1 2]));
Y0_norm = sqrt(sum(Y0.^2,[1 2]));

npcc = -sum(X0.*Y0,[1 2])./(X0_norm.*Y0_norm);
loss = mean(npcc,'all');

end

