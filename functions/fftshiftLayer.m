classdef fftshiftLayer < nnet.layer.Layer
  
    methods
        function layer = fftshiftLayer(name)
            % (Optional) Create a myLayer.
            % This function must have the same name as the class.

            % Set layer name.
            layer.Name = name;
            
            % Set layer description.
            layer.Description = "2-D fftshift layer";
            
          
        end
        
        function Z = predict(~, X)
            % Forward input data through the layer at prediction time and
            % output the result.
            %
            % Inputs:
            %         layer       - Layer to forward propagate through
            %         X1, ..., Xn - Input data
            % Outputs:
            %         Z1, ..., Zm - Outputs of layer forward function
            
            % Layer forward function for prediction goes here.
           
            Z = fftshift(fftshift(X,1),2);
            
        end

        function dLdX = backward(layer, X, ~, dLdZ, ~)
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

            dLdX = fftshift(fftshift(dLdZ,1),2);

        end
    end
end