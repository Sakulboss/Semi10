#l; conv2d; (1,16); (3,3); 1; 1;; a; relu;; p; maxpool; (3,3); 1; 1;; l; conv2d; (16, 32); (3,3); 1; 1;; v; view;; l; linear; (204800,5);;

r"""
l; layertype;        channels (in, out); kernel_size (h, w); stride; padding;;
l; conv2d, linear;   (3,3);             (3,3);               1;      1;;
p; pooltype;         size (h, w);       stride;              padding;;
p; avgpool, maxpool; (2,2);             1;                   0;;
a; activation_funtion;;
a; sigmoid, relu, tanh;;
v; view;;
"""


class NetStruct:
    def __init__(self):
        self.start_layers = ["- "]
        self.conv_sizes  :list[tuple] = [ (3,3), (11,21)] #[(1,1), (2,2), (3,3), (4,4), (5,5), (6,6), (7,7), (8,8), (9,9), (10,20), (10,50), (10,70)]
        self.pool_sizes  :list[tuple] = [(3,3), (5,5)]
        self.pool_types  :list[str]   = ["avgpool", "maxpool"]
        self.act_types   :list[str]   = [None, 'relu'] # 'tanh'
        self.dim         :list[int]   = [1,64,100] #Channel, Height, Width
        self.output_dim  :int         = 2 #Output Dimension
        self.linear_dim  :int         = 10 #Number of Linear Neurons
        self.filters     :list[tuple] = [(1,16),(16,48),(48,48)] #Number of channels
        self.layers      :list[str]   = []

        self.generator()
        self.save_net()

    def conv(self, layer:list, channel, stride: int = 1):
        conv_list = []
        for start_str in layer:
            for kernel in self.conv_sizes:
                padding = (int((kernel[0] - 1)/2), int((kernel[1] - 1)/2))
                conv_list.append(start_str + f'l; conv2d; {channel}; {kernel}; {stride}; {padding};; ')
        return self.pooling(self.activation(conv_list))

    def pooling(self, layer:list, stride=1):
        pool_list = []
        for start_str in layer:
            for p_type in self.pool_types:
                for size in self.pool_sizes:
                    padding = (int((size[0]-1)/2), int((size[1]-1)/2))
                    pool_list.append(start_str + f'p; {p_type}; {size}; {stride}; {padding};; ')
        return pool_list

    def first_linear(self, layer:list, input_filters:int):
        linear_list, size = [], input_filters * self.dim[1] * self.dim[2]
        kernel_size = (size, self.linear_dim)
        for start_str in layer:
                linear_list.append(start_str + f'l; linear; {kernel_size};; ')
        return self.activation(linear_list)

    def middle_linear(self, layer:list):
        linear_list = []
        kernel_size = (self.linear_dim, self.linear_dim)
        for start_str in layer:
            linear_list.append(start_str + f'l; linear; {kernel_size};; ')
        return self.activation(linear_list)

    def last_linear(self, layer:list):
        last_list = []
        kernel_size = (self.linear_dim, self.output_dim)
        for start_str in layer:
                last_list.append(start_str + f'l; linear; {kernel_size};; ')
        return last_list

    def linear_only(self, layer:list, input_filters:int):
        linear_list = []
        kernel_size = (input_filters * self.dim[1] * self.dim[2], self.output_dim)
        for start_str in layer:
            linear_list.append(start_str + f'l; linear; {kernel_size};; ')
        return self.activation(linear_list)

    def activation(self, layer:list):
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
        with open('_netstruct1.txt', 'w') as f:
            for layer in self.layers:
                f.write(layer + '\n')
        print('Netzstruktur gespeichert: {} Modelle.'.format(len(self.layers)))


if __name__=='__main__':
    # Example usage
    # Create an instance of the NetStruct class
    # This will generate the network structure and save it to a file
    Cnn = NetStruct()
