classdef plusLayer < nnet.layer.Layer
    
    properties
        % (Optional) Layer properties.
        
        % Layer properties go here.
        Addend        
    end

    methods
        function layer = plusLayer(A,name)
            % (Optional) Create a myLayer.
            % This function must have the same name as the class.
            
            % Set layer name.
            layer.Name = name;
            
            % Set layer description.
            layer.Description = "Plus a known number";
            
            % Set multiplier.
            layer.Addend = A;
            
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
            A = layer.Addend;
            Z = X + A;
        end

    end
end