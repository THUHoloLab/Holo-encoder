clearvars;
close all;clc

addpath('./functions');

lgraph = layerGraph;
inputSize = [2160 3840];
outputSize = [3840 3840];

layers = [
    imageInputLayer([inputSize 1],'Normalization','none','Name','input')];

lgraph = addLayers(lgraph,layers);

[lgraph,outputName] = unet(lgraph, 'input');

% odd to even
% lgraph = replaceLayer(lgraph,'ResUpConv3_1',...
%     transposedConv2dLayer(3,64,'Stride',2,'Cropping',[1 1 0 1],'Name','ResUpConv3_1'));
% lgraph = replaceLayer(lgraph,'SkipUpConv3',...
%     transposedConv2dLayer(3,64,'Stride',2,'Cropping',[1 1 0 1],'Name','SkipUpConv3'));

layers = [
    batchNormalizationLayer('Name','BN')
    tanhpiLayer('tanh')];

lgraph = addLayers(lgraph,layers);
lgraph = connectLayers(lgraph,outputName,'BN');

lambda = 532e-6;
z = 160;
dp = 0.00374;
Lx = dp*inputSize(2);
Ly = dp*inputSize(1);
[x,y] = meshgrid(-Lx/2:dp:Lx/2-dp,-Ly/2:dp:Ly/2-dp);
P = pi*(x.^2 + y.^2)/(lambda*z);

lgraph = addLayers(lgraph,fresnelLayer(inputSize,outputSize,P,'Name','I'));
lgraph = connectLayers(lgraph,'tanh','I');

dlnet = dlnetwork(lgraph);
save('HoloEncoder_untrained.mat','dlnet');
