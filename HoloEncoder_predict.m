clearvars -except dlnet;
close all;
clc;

addpath('./functions');
load HoloEncoder_trained.mat
    
X = imread('./images/0805.png');
X = im2gray(X);
X = imresize(X,[2160,3840]);
X = single(X);
[m,n] = size(X);
dlX = gpuArray(dlarray(X,'SSCB'));
tic
dlY = forward(dlnet,dlX,'Outputs','tanh');
toc;
dlZ = forward(dlnet,dlX);

Y = gather(extractdata(dlY));
Z = gather(extractdata(dlZ));
figure,imshow(Y,[]);title('hologram')
figure,imshow(Z,[]);title('reconstruction')
