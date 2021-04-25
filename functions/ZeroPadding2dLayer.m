classdef ZeroPadding2dLayer < nnet.layer.Layer
    % ZeroPadding2dLayer   ZeroPadding2dLayer layer
    %
    %   layer = ZeroPadding2dLayer(Name, Amounts) creates a layer with
    %   name Name that pads the input image with zeros.
    %
    %       layer = ZeroPadding2dLayer(Name, Pad), where Pad is a scalar
    %           integer, pads the top, left, bottom and right with the same
    %           number of rows (and columns).
    %
    %       layer = ZeroPadding2dLayer(Name, [Vertical, Horizontal]),
    %           pads the top and bottom with 'Vertical' rows, and the left
    %           and right with 'Horizontal' columns.
    %
    %       layer = ZeroPadding2dLayer(Name, [Top, Bottom, Left, Right]),
    %           pads the top, bottom, left and right with the specified
    %           numbers of rows and columns.
    
    %   Copyright 2017 The MathWorks, Inc.
    properties
        Top
        Bottom
        Left
        Right
    end
    
    methods
        function this = ZeroPadding2dLayer(name, Amounts)
            assert(all(Amounts >= 0));
            if isscalar(Amounts)
                this.Top    = Amounts;
                this.Bottom = Amounts;
                this.Left   = Amounts;
                this.Right  = Amounts;
            elseif isvector(Amounts)
                if numel(Amounts)==2
                    this.Top    = Amounts(1);
                    this.Bottom = Amounts(1);
                    this.Left   = Amounts(2);
                    this.Right  = Amounts(2);
                elseif numel(Amounts)==4
                    this.Top    = Amounts(1);
                    this.Bottom = Amounts(2);
                    this.Left   = Amounts(3);
                    this.Right  = Amounts(4);
                end
            else
                throwAsCaller(MException(message('nnet_cnn_kerasimporter:keras_importer:ZP2DAmounts')));
            end
            this.Name = name;
%             this.Description = getString(message('nnet_cnn_kerasimporter:keras_importer:ZeroPadding2dDescription'));
%             this.Type = getString(message('nnet_cnn_kerasimporter:keras_importer:ZeroPadding2dType'));
        end
        
        function Z = predict( this, X )
            % X is size [H W C N]. Z is size [H+Top+Bottom, W+Left+Right, C, N].
            [H,W,C,N] = size(X);
            Z = zeros(H + this.Top + this.Bottom, W + this.Left + this.Right, C, N, 'like', X);
            Z(this.Top+(1:H), this.Left+(1:W), :, :) = X;
        end
        
%         function dLdX = backward( this, X, Z, dLdZ, memory )
%             % dLdX and X are size [H W C N]. dLdZ is size [H+Top+Bottom, W+Left+Right, C, N]. 
%             [H,W,C,N] = size(X);
%             dLdX = dLdZ(this.Top+(1:H), this.Left+(1:W), :, :);
%         end
    end
end
