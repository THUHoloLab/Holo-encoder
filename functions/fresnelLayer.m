classdef fresnelLayer < nnet.layer.Layer %#codegen
   
    properties
        % imageConv block.
        Network
    end
    
    methods
        function layer = fresnelLayer(inputSize,outputSize,P,NameValueArgs)
            
            % Parse input arguments.
            arguments
                inputSize
                outputSize
                P
                NameValueArgs.Name = ''
            end
            name = NameValueArgs.Name;

            % Set number of inputs.
            layer.NumInputs = 1;
            
            % Set layer name.
            layer.Name = name;
            
            % Set layer description.
            layer.Description = "Fresnel proagation of phase-only input";
           
            % Define nested layer graph.
            lgraph = layerGraph;
            layers = [
                imageInputLayer([inputSize 1],'Normalization','None','Name','in')
                plusLayer(P,'plus')];
            lgraph = addLayers(lgraph,layers);
                     
            lgraph = addLayers(lgraph,cosLayer('cos'));
            lgraph = addLayers(lgraph,sinLayer('sin'));
            lgraph = connectLayers(lgraph,'plus','cos');
            lgraph = connectLayers(lgraph,'plus','sin');
            
            % zero padding
            lgraph = addLayers(lgraph,ZeroPadding2dLayer('cospad', (outputSize - inputSize)/2));
            lgraph = addLayers(lgraph,ZeroPadding2dLayer('sinpad', (outputSize - inputSize)/2));
            lgraph = connectLayers(lgraph,'cos','cospad');
            lgraph = connectLayers(lgraph,'sin','sinpad');
            
            % fftshift
            lgraph = addLayers(lgraph,fftshiftLayer('cosshift'));
            lgraph = addLayers(lgraph,fftshiftLayer('sinshift'));
            lgraph = connectLayers(lgraph,'cospad','cosshift');
            lgraph = connectLayers(lgraph,'sinpad','sinshift');
            
            % fft2
            lgraph = addLayers(lgraph,fft2DLayer('Fcos'));
            lgraph = addLayers(lgraph,fft2DLayer('Fsin'));
            lgraph = connectLayers(lgraph,'cosshift','Fcos');
            lgraph = connectLayers(lgraph,'sinshift','Fsin');
            
            lgraph = addLayers(lgraph,subtractionLayer('Fr'));
            lgraph = addLayers(lgraph,additionLayer(2,'Name','Fi'));
            lgraph = connectLayers(lgraph,'Fcos/real','Fr/in1');
            lgraph = connectLayers(lgraph,'Fsin/imag','Fr/in2');
            lgraph = connectLayers(lgraph,'Fcos/imag','Fi/in1');
            lgraph = connectLayers(lgraph,'Fsin/real','Fi/in2');
                                   
            % intensity
            lgraph = addLayers(lgraph,intensityLayer('I'));
            lgraph = connectLayers(lgraph,'Fr','I/in1');
            lgraph = connectLayers(lgraph,'Fi','I/in2');
            
            % fftshift
            lgraph = addLayers(lgraph,fftshiftLayer('fftshift2'));
            lgraph = connectLayers(lgraph,'I','fftshift2');
            
            % crop
%             lgraph = addLayers(lgraph,crop2dLayer('centercrop','Name','crop'));
%             lgraph = connectLayers(lgraph,'fftshift2','crop/in');
%             lgraph = connectLayers(lgraph,'in','crop/ref');
            
            % Convert to dlnetwork.
            dlnet = dlnetwork(lgraph);
    
            % Set Network property.
            layer.Network = dlnet;
            
        end
        
        function Z = predict(layer, X)
            % Forward input data through the layer at prediction time and
            % output the result.
            %
            % Inputs:
            %         layer       - Layer to forward propagate through
            %         X1, ..., Xn - Input data
            % Outputs:
            %         Z1, ..., Zm - Outputs of layer forward function
            
            % Layer forward function for prediction goes here.
            
            X = dlarray(X,'SSCB');
            
            % Predict using network.
            dlnet = layer.Network;
            Z = predict(dlnet,X);
            
            % Strip dimension labels.
            Z = stripdims(Z);
            
        end

    end
end