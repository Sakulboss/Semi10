#l; conv; (1,16); (3,3); 1; 1;; a; relu;; p; maxpool; (3,3); 1; 1;; l; conv; (16, 32); (3,3); 1; 1;; v; view;; l; linear; (204800,5);;
from tensorflow.python.autograph.operators import new_list


class NetStruct:
    def __init__(self):
        self.start_layers = ["- "]
        self.conv_sizes  :list[tuple] = [ (3,3), (10,20)] #[(1,1), (2,2), (3,3), (4,4), (5,5), (6,6), (7,7), (8,8), (9,9), (10,20), (10,50), (10,70)]
        self.pool_sizes  :list[tuple] = [(2,2), (3,3)]
        self.pool_types  :list[str]   = ["avgpool", "maxpool"]
        self.act_types   :list[str]   = [None, 'sigmoid', 'relu', 'tanh']
        self.conv_layers :list[int]   = [1,2]
        self.lin_layers  :list[int]   = [0,1]
        self.dim         :list[int]   = [1,60,100] #Channel, Height, Width
        self.output_dim  :int         = 2 #Output Dimension
        self.filters     :list[tuple] = [(1,16),(16,48),(48,48)] #Number of channels
        self.layers      :list[str]   = []
        self.generator()
        self.save_net()


    def conv(self, layer:list, channel, stride: int = 1):
        conv_list = []
        for start_str in layer:
            for kernel in self.conv_sizes:
                padding = (kernel[0] - 1, kernel[1] - 1)
                conv_list.append(start_str + f'l; conv2d; {channel}; {kernel}; {stride}; {padding};; ')
        return self.pooling(self.activation(conv_list))

    def pooling(self, layer:list, stride=1):
        pool_list = []
        for start_str in layer:
            for p_type in self.pool_types:
                for size in self.pool_sizes:
                    padding = (size[0]-1, size[1]-1)
                    pool_list.append(start_str + f'p; {p_type}; {size}; {stride}; {padding};; ')
        return pool_list

    def linear(self, layer:list, input_filters:int):
        linear_list, size = [], input_filters * self.dim[1] * self.dim[2]
        kernel_size = (size, size)
        for start_str in layer:
                linear_list.append(start_str + f'l; linear; {kernel_size};; ')
        return self.activation(linear_list)

    def last_layer(self, layer:list, input_filters:int):
        last_list = []
        kernel_size = (input_filters * self.dim[1] * self.dim[2], self.output_dim)
        for start_str in layer:
                last_list.append(start_str + f'l; linear; {kernel_size};; ')
        return last_list

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

        #3 Conv, 3 Linear
        self.layers += self.last_layer(
            self.linear(
                self.linear(
                    self.view_layer(
                        self.conv(
                            self.conv(
                                self.conv(
                                    self.start_layers, self.filters[0]
                                ), self.filters[1]
                            ), self.filters[2]
                        )
                    ), self.filters[2][1]
                ), self.filters[2][1]
            ), self.filters[2][1]
        )

        #2 Conv, 3 Linear
        self.layers += self.last_layer(
            self.linear(
                self.linear(
                    self.view_layer(
                        self.conv(
                            self.conv(
                                    self.start_layers, self.filters[0]
                            ), self.filters[1]
                        )
                    ), self.filters[1][1]
                ), self.filters[1][1]
            ), self.filters[1][1]
        )

        #1 Conv, 3 Linear
        self.layers += self.last_layer(
            self.linear(
                self.linear(
                    self.view_layer(
                        self.conv(
                            self.start_layers, self.filters[0]
                        )
                    ), self.filters[0][1]
                ), self.filters[0][1]
            ), self.filters[0][1]
        )

        #3 Conv, 2 Linear
        self.layers += self.last_layer(
            self.linear(
                    self.view_layer(
                        self.conv(
                            self.conv(
                                self.conv(
                                    self.start_layers, self.filters[0]
                                ), self.filters[1]
                            ), self.filters[2]
                        )
                ), self.filters[2][1]
            ), self.filters[2][1]
        )

        #3 Conv, 1 Linear
        self.layers += self.last_layer(
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
        self.layers += self.last_layer(
            self.linear(
                self.view_layer(
                    self.conv(
                        self.conv(
                            self.start_layers, self.filters[0]
                        ), self.filters[1]
                    )
                ), self.filters[1][1]
            ), self.filters[1][1]
        )

        #2 Conv, 1 Linear
        self.layers += self.last_layer(
            self.view_layer(
                self.conv(
                    self.conv(
                        self.start_layers, self.filters[0]
                    ), self.filters[1]
                )
            ), self.filters[1][1]
        )

        #1 Conv, 2 Linear
        self.layers += self.last_layer(
            self.linear(
                self.view_layer(
                    self.conv(
                        self.start_layers, self.filters[0]
                    )
                ), self.filters[0][1]
            ), self.filters[0][1]
        )

        #1 Conv, 1 Linear
        self.layers += self.last_layer(
                self.view_layer(
                        self.conv(
                            self.start_layers, self.filters[0]
                    )
            ), self.filters[0][1]
        )

    def save_net(self):
        with open('netstruct.txt', 'w') as f:
            for layer in self.layers:
                f.write(layer + '\n')
        print('Netzstruktur gespeichert.')

Cnn = NetStruct()
