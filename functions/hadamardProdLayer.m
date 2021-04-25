classdef hadamardProdLayer < nnet.layer.Layer
    
    properties
        % (Optional) Layer properties.
        
        % Layer properties go here.
        Multiplier        
    end

    methods
        function layer = hadamardProdLayer(H,name)
            % (Optional) Create a myLayer.
            % This function must have the same name as the class.
            
            % Set layer name.
            layer.Name = name;
            
            % Set layer description.
            layer.Description = "Hadamard Product of two same size array";
            
            % Set multiplier.
            layer.Multiplier = H;
            
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
            H = layer.Multiplier;
            Z = X.*H;
        end

    end
end