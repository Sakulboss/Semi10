
r"""
This script generates a neural network structure for a convolutional neural network in a self-made format after the following structure:
l; layer type;          channels (in, out); kernel_size (h, w); stride; padding;;
-> l; conv2d;             (3,3);             (3,3);               1;      1;;
-> l; linear;             (204800,2);;
p; pool type;           size (h, w);       stride;              padding;;
-> p; avgpool, maxpool;   (2,2);             1;                   0;;
a; activation functions;;
-> a; sigmoid, relu, tanh;;
v; view layer, converts 2D plane from the convolutional layers to 1D plane for the hidden layers;;
-> v; view;;

Example:
l; conv2d; (1,16); (3,3); 1; 1;; a; relu;; p; maxpool; (3,3); 1; 1;; l; conv2d; (16, 32); (3,3); 1; 1;; v; view;; l; linear; (204800,2);;
"""


class NetStruct:
    def __init__(self, used_layers:int = 0):
        """
        This class will generate the structure with the following parameters:
        """
        self.start_layers = ["- "]                                  # Start characters of the layer
        self.conv_sizes  :list[tuple] = [ (3,3), (11,21)]           # convolutional kernel sizes
        self.pool_sizes  :list[tuple] = [(3,3), (5,5)]              # pooling kernel sizes
        self.pool_types  :list[str]   = ["avgpool","maxpool"]       # pooling types
        self.act_types   :list[str]   = [None, "relu"]              # activation functions
        self.dim         :list[int]   = [1,64,100]                  # dimensions of the tensor in the CNN -> Channel, Height, Width
        self.output_dim  :int         = 2                           # number of output dimensions
        self.linear_dim  :int         = 10                          # number of linear neurons after the first linear layer
        self.filters     :list[tuple] = [(1,16),(16,48),(48,48)]    # Number of filters in the convolutional layers
        self.layers      :list[str]   = []                          # the layers will be saved here

        self.generator()                # generate the network structure
        self.used_layers(used_layers)   # set the trained layers
        self.save_net()                 # save the network structure to a file

    def conv(self, layer:list, channel, stride: int = 1):
        """
        Create a convolutional layer with the given parameters.
        Args:
            layer:   the created layers, to each will be all possibilities of this layer added
            channel: number of channels in the layer, defines the number of filters
            stride:  movement of the filter, here set to the standard (1)

        Returns:
            list: list with all possible layers
        """
        conv_list = []
        for start_str in layer:
            for kernel in self.conv_sizes:
                padding = (int((kernel[0] - 1)/2), int((kernel[1] - 1)/2))
                conv_list.append(start_str + f'l; conv2d; {channel}; {kernel}; {stride}; {padding};; ')
        return self.pooling(self.activation(conv_list))

    def pooling(self, layer:list, stride=1):
        """
        Create a pooling layer with the given parameters.
        Args:
            layer:  the created layers, to each will be all possibilities of this layer added
            stride: movement of the filter, here set to 1

        Returns:
            list: list with all possible layers
        """
        pool_list = []
        for start_str in layer:
            for p_type in self.pool_types:
                for size in self.pool_sizes:
                    padding = (int((size[0]-1)/2), int((size[1]-1)/2))
                    pool_list.append(start_str + f'p; {p_type}; {size}; {stride}; {padding};; ')
        return pool_list

    def first_linear(self, layer:list, input_filters:int):
        """
        Create the first linear layer if more than one is used with the given parameters. There are self.linear_dim exit neurons.
        Args:
            layer:          the created layers, to each will be all possibilities of this layer added
            input_filters:  number of input filters to calculate the input size of the linear layer
        Returns:
            list: list with all possible layers
        """
        linear_list, size = [], input_filters * self.dim[1] * self.dim[2]
        kernel_size = (size, self.linear_dim)
        for start_str in layer:
                linear_list.append(start_str + f'l; linear; {kernel_size};; ')
        return self.activation(linear_list)

    def middle_linear(self, layer:list):
        """
        Create the middle linear layer if three linear layers are used with the given parameters. There are self.linear_dim entry and exit neurons.
        Args:
            layer:          the created layers, to each will be all possibilities of this layer added
        Returns:
            list: list with all possible layers
        """
        linear_list = []
        kernel_size = (self.linear_dim, self.linear_dim)
        for start_str in layer:
            linear_list.append(start_str + f'l; linear; {kernel_size};; ')
        return self.activation(linear_list)

    def last_linear(self, layer:list):
        """
        Create the last linear layer if more than one linear layer is used with the given parameters. There are self.linear_dim entry and self.output_dim exit neurons.
        Args:
            layer:  the created layers, to each will be all possibilities of this layer added
        Returns:
            list: list with all possible layers
        """
        last_list = []
        kernel_size = (self.linear_dim, self.output_dim)
        for start_str in layer:
                last_list.append(start_str + f'l; linear; {kernel_size};; ')
        return last_list

    def linear_only(self, layer:list, input_filters:int):
        """
        Create a special linear layer if only one is used with the given parameters. There are self.output_dim exit neurons.
        Args:
            layer:          the created layers, to each will be all possibilities of this layer added
            input_filters:  number of input filters to calculate the input size of the linear layer
        Returns:
            list: list with all possible layers
        """
        linear_list = []
        kernel_size = (input_filters * self.dim[1] * self.dim[2], self.output_dim)
        for start_str in layer:
            linear_list.append(start_str + f'l; linear; {kernel_size};; ')
        return self.activation(linear_list)

    def activation(self, layer:list):
        """
        Create an activation layer with the given parameters.
        Args:
            layer: the created layers, to each will be all possibilities of this layer added
        Returns:
            list: list with all possible layers
        """
        act_list = []
        for start_str in layer:
            for activation in self.act_types:
                if activation is not None:
                    act_list.append(start_str + f'a; {activation};; ')
                else:
                    act_list.append(start_str)
        return act_list

    @staticmethod
    def view_layer(layer:list):
        """
        Create a view layer with the given parameters.
        Args:
            layer: the created layers, to each will be all possibilities of this layer added
        Returns:
            list: list with all possible layers
        """
        view_list = []
        for start_str in layer:
            view_list.append(start_str + f'v: view;; ')
        return view_list

    def generator(self):

        #3 Conv, 3 Linear --
        self.layers += self.last_linear(
            self.middle_linear(
                self.first_linear(
                    self.view_layer(
                        self.conv(
                            self.conv(
                                self.conv(
                                    self.start_layers, self.filters[0]
                                ), self.filters[1]
                            ), self.filters[2]
                        )
                    ), self.filters[2][1]
                )
            )
        )

        #2 Conv, 3 Linear
        self.layers += self.last_linear(
            self.middle_linear(
                self.first_linear(
                    self.view_layer(
                        self.conv(
                            self.conv(
                                    self.start_layers, self.filters[0]
                            ), self.filters[1]
                        )
                    ), self.filters[1][1]
                )
            )
        )

        #1 Conv, 3 Linear
        self.layers += self.last_linear(
            self.middle_linear(
                self.first_linear(
                    self.view_layer(
                        self.conv(
                            self.start_layers, self.filters[0]
                        )
                    ), self.filters[0][1]
                )
            )
        )

        #3 Conv, 2 Linear
        self.layers += self.last_linear(
            self.first_linear(
                    self.view_layer(
                        self.conv(
                            self.conv(
                                self.conv(
                                    self.start_layers, self.filters[0]
                                ), self.filters[1]
                            ), self.filters[2]
                        )
                ), self.filters[2][1]
            )
        )

        #3 Conv, 1 Linear
        self.layers += self.linear_only(
            self.view_layer(
                self.conv(
                    self.conv(
                        self.conv(
                            self.start_layers, self.filters[0]
                        ), self.filters[1]
                    ), self.filters[2]
                )
            ), self.filters[2][1]
        )

        #2 Conv, 2 Linear
        self.layers += self.last_linear(
            self.first_linear(
                self.view_layer(
                    self.conv(
                        self.conv(
                            self.start_layers, self.filters[0]
                        ), self.filters[1]
                    )
                ), self.filters[1][1]
            )
        )

        #2 Conv, 1 Linear
        self.layers += self.linear_only(
            self.view_layer(
                self.conv(
                    self.conv(
                        self.start_layers, self.filters[0]
                    ), self.filters[1]
                )
            ), self.filters[1][1]
        )

        #1 Conv, 2 Linear
        self.layers += self.last_linear(
            self.first_linear(
                self.view_layer(
                    self.conv(
                        self.start_layers, self.filters[0]
                    )
                ), self.filters[0][1]
            )
        )

        #1 Conv, 1 Linear
        self.layers += self.linear_only(
                self.view_layer(
                        self.conv(
                            self.start_layers, self.filters[0]
                    )
            ), self.filters[0][1]
        )

    def save_net(self):
        with open('_netstruct.txt', 'w') as f:
            for layer in self.layers:
                f.write(layer + '\n')
        print('Netzstruktur gespeichert: {} Modelle.'.format(len(self.layers)))

    def used_layers(self, trained_layers:int = 0):
        """
         Setting the trained layers if training happened earlier
        """
        self.layers = [ '#' + self.layers[i][1:] if i + 1 <= trained_layers else self.layers[i] for i in range(len(self.layers))]
        return self.layers

if __name__=='__main__':
    # Example usage:
    # Create an instance of the NetStruct class
    # This will generate the network structure and save it to a file
    Cnn = NetStruct(5179)
