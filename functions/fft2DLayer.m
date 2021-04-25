classdef fft2DLayer < nnet.layer.Layer
    
    methods
        function layer = fft2DLayer(name)
            % (Optional) Create a myLayer.
            % This function must have the same name as the class.

            % Set layer name.
            layer.Name = name;
            
            % Set number of outputs.
            layer.NumOutputs = 2;
            
            % Set output names.
            layer.OutputNames = {'real','imag'};
            
            % Set layer description.
            layer.Description = "2-D Fourier transform layer";
            
        end
        
        function [Z1,Z2] = predict(layer, X)
            % Forward input data through the layer at prediction time and
            % output the result.
            %
            % Inputs:
            %         layer       - Layer to forward propagate through
            %         X1, ..., Xn - Input data
            % Outputs:
            %         Z1, ..., Zm - Outputs of layer forward function
            
            % Layer forward function for prediction goes here.
           
            Z = fft2(X);
            Z1 = real(Z);
            Z2 = imag(Z);
        end
        
        function dLdX = backward(layer, X, ~,~, dLdZ1,dLdZ2, ~)
            % Backward propagate the derivative of the loss function through 
            % the layer.
            %
            % Inputs:
            %         layer             - Layer to backward propagate through
            %         X1, ..., Xn       - Input data
            %         Z1, ..., Zm       - Outputs of layer forward function            
            %         dLdZ1, ..., dLdZm - Gradients propagated from the next layers
            %         memory            - Memory value from forward function
            % Outputs:
            %         dLdX1, ..., dLdXn - Derivatives of the loss with respect to the
            %                             inputs
            %         dLdW1, ..., dLdWk - Derivatives of the loss with respect to each
            %                             learnable parameter
            
            % Layer backward function goes here.

            dLdX1 = real(fft2(dLdZ1));
            dLdX2 = imag(fft2(dLdZ2));
            dLdX = dLdX1 + dLdX2;

        end
       
    end
end