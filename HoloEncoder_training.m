clearvars  -except dlnet averageGrad averageSqGrad;
close all; clc

addpath('./functions');

%% data
% load network
load HoloEncoder_untrained.mat

% load dataset
rawImagePath = 'F:\DIV2K_valid_HR';
% rawImagePath = '/home/user/Jason/Dataset/ImageNet_20121';
imds = imageDatastore(rawImagePath,'IncludeSubfolders',true);
augimds = augmentedImageDatastore([2160 3840],imds,'ColorPreprocessing',"rgb2gray");

% initialize plot
[ax1,ax2,lineLossTotal]=initializePlots();
plotFrequency = 5;

%% training parameters
numEpochs = 5;
miniBatchSize = 1;
augimds.MiniBatchSize = miniBatchSize;
averageGrad = [];
averageSqGrad = [];
numIterations = floor(augimds.NumObservations*numEpochs/miniBatchSize);

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
    reset(augimds);
    augimds = shuffle(augimds);
    
    % Loop over mini-batches.
    while hasdata(augimds)
%         iteration = iteration + 1;
                
        % Read mini-batch of data.
        data = read(augimds);
                       
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

        for iteration = 1:100
        % Evaluate model gradients.
        [gradients,dlYc,loss] = dlfeval(@modelGradients,dlnet,dlX);

        % Update the network parameters using the Adam optimizer.
        [dlnet,averageGrad,averageSqGrad] = ...
            adamupdate(dlnet,gradients,averageGrad,averageSqGrad,iteration,...
            learnRate, gradientDecayFactor, squaredGradientDecayFactor);
        

        addpoints(lineLossTotal,iteration,double(gather(extractdata(loss))))
%         addpoints(lineLossPercept,iteration,double(gather(extractdata(lossPercept))))
%         addpoints(lineLossNpcc,iteration,double(gather(extractdata(lossNpcc))))
        
        % Every plotFequency iterations, plot the training progress.
        if mod(iteration,plotFrequency) == 0            
            % Use the first image of the mini-batch as a validation image.
            dlV = dlX(:,:,:,1);
            % Use the transformed validation image computed previously.
            dlVY = dlYc(:,:,:,1);
            dlVY = rescale(dlVY,0,255);
            dlZ = forward(dlnet,dlX,'Outputs','tanh');
            dlVZ = dlZ(:,:,:,1);
            dlVZ = rescale(dlVZ,0,255);
            
            % To use the function imshow, convert to uint8.
            validationImage = uint8(gather(extractdata(dlV)));
            transformedValidationImage = uint8(gather(extractdata(dlVY)));
            phaseImage = uint8(gather(extractdata(dlVZ)));
            
            % Plot the input image and the output image and increase size
            imshow(imtile({validationImage,transformedValidationImage,phaseImage},'GridSize', [1 3]),'Parent',ax2);
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

save('Unet_asm_trained.mat','dlnet','averageGrad','averageSqGrad');

%% Prediction

X = imread('D:\Dataset\DIV2K_valid_HR\0801.png');
X = rgb2gray(X);
X = imresize(X,[512 512]);
X = single(X);
dlX = dlarray(X,'SSCB');

[dlZ,dlY] = forward(dlnet,dlX,'Outputs',{'tanh','I'});
Y = extractdata(dlY);
Z = extractdata(dlZ);
figure,imshow(Y,[]);
figure,imshow(Z,[]);

%% netloss

function [gradients,dlYc,loss] = modelGradients(dlnet,dlX)

    dlY = forward(dlnet,dlX);
    dlYc = dlY(853:1196,853:1196,:,:);
    X = gather(extractdata(dlX));
    Xc = imresize(X,[344 344]);
    Xc = padarray(Xc,[852 852]);
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

