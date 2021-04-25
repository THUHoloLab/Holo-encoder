function [newlgraph,outputName] = unet(lgraph, inputName)

[lgraph,outputName] = ResDownBlock(lgraph, inputName,'FilterSize',9,'numFilters',16,'DilationFactor',3, 'Order',1);
[lgraph,outputName] = ResDownBlock(lgraph,outputName,'FilterSize',3,'numFilters',32,'DilationFactor',2, 'Order',2);
[lgraph,outputName] = ResDownBlock(lgraph,outputName,'FilterSize',3,'numFilters',64,'DilationFactor',1, 'Order',3);
[lgraph,outputName] = ResDownBlock(lgraph,outputName,'FilterSize',3,'numFilters',96,'DilationFactor',1, 'Order',4);
[lgraph,outputName] = ResUpBlock(lgraph,outputName,'FilterSize',3,'numFilters',64,'Order',3);
[lgraph,outputName] = ResUpBlock(lgraph,outputName,'FilterSize',3,'numFilters',32,'Order',2);
[lgraph,outputName] = ResUpBlock(lgraph,outputName,'FilterSize',3,'numFilters',16,'Order',1);
[lgraph,outputName] = ResUpBlock(lgraph,outputName,'FilterSize',3,'numFilters',1,'Order',0,'NumInputs',2);
% [lgraph,outputName] = ResUpBlock(lgraph,outputName,'FilterSize',3,'numFilters',1,'Order',10,'NumInputs',2);

% skip connection
% lgraph = connectLayers(lgraph,inputName,'ResUAdd0/in3');
lgraph = connectLayers(lgraph,'ResDAdd1','ResUAdd1/in3');
lgraph = connectLayers(lgraph,'ResDAdd2','ResUAdd2/in3');
lgraph = connectLayers(lgraph,'ResDAdd3','ResUAdd3/in3');
% lgraph = connectLayers(lgraph,'ResDAdd4','ResUAdd4/in3');

newlgraph = lgraph;
end

%% Residual Downsampling Block

function [newlgraph,outputName] = ResDownBlock(lgraph,inputName,NameValueArgs)

arguments
    lgraph
    inputName
    NameValueArgs.NumFilters
    NameValueArgs.FilterSize = 3;
    NameValueArgs.DilationFactor = 1
    NameValueArgs.Order
end
numFilters = NameValueArgs.NumFilters;
filterSize = NameValueArgs.FilterSize;
dilationFactor = NameValueArgs.DilationFactor;
order = NameValueArgs.Order;         

k = num2str(order);
Layers = [
    batchNormalizationLayer('Name',['ResDownBN' k '_1'])
    reluLayer('Name',['ResDownReLU' k '_1'])
    convolution2dLayer(filterSize,numFilters, 'DilationFactor',dilationFactor,'Stride',2,'Padding','same','Name',['ResDownConv' k '_1'])
    batchNormalizationLayer('Name',['ResDownBN' k '_2'])
    reluLayer('Name',['ResDownReLU' k '_2'])
    convolution2dLayer([3 3],numFilters, 'Padding','same','Name',['ResDownConv' k '_2'])
    additionLayer(2,'Name',['ResDownAdd' k])
    
    batchNormalizationLayer('Name',['ResDBN' k '_1'])
    reluLayer('Name',['ResDReLU' k '_1'])
    convolution2dLayer([3 3],numFilters,'Stride',[1 1],'Padding','same','Name',['ResDConv' k '_1'])
    batchNormalizationLayer('Name',['ResDBN' k '_2'])
    reluLayer('Name',['ResDReLU' k '_2'])
    convolution2dLayer([3 3],numFilters,'Stride',[1 1],'Padding','same','Name',['ResDConv' k '_2'])
    additionLayer(2,'Name',['ResDAdd' k])];

lgraph = addLayers(lgraph,Layers);
lgraph = connectLayers(lgraph,inputName,['ResDownBN' k '_1']);

skip = convolution2dLayer([1 1],numFilters,'Stride',[2 2],'Padding','same','Name',['SkipDownConv' k]);
lgraph = addLayers(lgraph,skip);

lgraph = connectLayers(lgraph,inputName,['SkipDownConv' k]);
lgraph = connectLayers(lgraph,['SkipDownConv' k],['ResDownAdd' k '/in2']);
newlgraph = connectLayers(lgraph,['ResDownAdd' k],['ResDAdd' k '/in2']);
outputName = ['ResDAdd' k];
end


%% Residual Upsampling Block

function [newlgraph,outputName] = ResUpBlock(lgraph,inputName,NameValueArgs)

arguments
    lgraph
    inputName
    NameValueArgs.NumFilters
    NameValueArgs.FilterSize = 3
    NameValueArgs.Stride = 1
    NameValueArgs.Order
    NameValueArgs.NumInputs = 3
end
numFilters = NameValueArgs.NumFilters;
filterSize = NameValueArgs.FilterSize;
order = NameValueArgs.Order;
numInputs = NameValueArgs.NumInputs;

k = num2str(order);
Layers = [
    batchNormalizationLayer('Name',['ResUpBN' k '_1'])
    reluLayer('Name',['ResUpReLU' k '_1'])
    transposedConv2dLayer(filterSize,numFilters,'Stride',2,'Cropping','same','Name',['ResUpConv' k '_1'])
    batchNormalizationLayer('Name',['ResUpBN' k '_2'])
    reluLayer('Name',['ResUpReLU' k '_2'])
    convolution2dLayer([3 3],numFilters,'Stride',[1 1],'Padding','same','Name',['ResUpConv' k '_2'])
    additionLayer(2,'Name',['ResUpAdd' k])
    
    batchNormalizationLayer('Name',['ResUBN' k '_1'])
    reluLayer('Name',['ResUReLU' k '_1'])
    convolution2dLayer([3 3],numFilters,'Stride',[1 1],'Padding','same','Name',['ResUConv' k '_1'])
    batchNormalizationLayer('Name',['ResUBN' k '_2'])
    reluLayer('Name',['ResUReLU' k '_2'])
    convolution2dLayer([3 3],numFilters,'Stride',[1 1],'Padding','same','Name',['ResUConv' k '_2'])
    additionLayer(numInputs,'Name',['ResUAdd' k])];

lgraph = addLayers(lgraph,Layers);
lgraph = connectLayers(lgraph,inputName,['ResUpBN' k '_1']);

skip = transposedConv2dLayer([2 2],numFilters,'Stride',[2 2],'Cropping','same','Name',['SkipUpConv' k]);
lgraph = addLayers(lgraph,skip);

lgraph = connectLayers(lgraph,inputName,['SkipUpConv' k]);
lgraph = connectLayers(lgraph,['SkipUpConv' k],['ResUpAdd' k '/in2']);
newlgraph = connectLayers(lgraph,['ResUpAdd' k],['ResUAdd' k '/in2']);
outputName = ['ResUAdd' k];

end
