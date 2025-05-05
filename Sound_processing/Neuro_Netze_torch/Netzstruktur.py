#l; conv; (1,16); (3,3); 1; 1;; a; relu;; p; maxpool; (3,3); 1; 1;; l; conv; (16, 32); (3,3); 1; 1;; v; view;; l; linear; (204800,5);;

class NetStruct:
    def __init__(self):
        self.layers = ["- "]
        self.conv_sizes  :list[tuple] = [ (3,3), (10,20)] #[(1,1), (2,2), (3,3), (4,4), (5,5), (6,6), (7,7), (8,8), (9,9), (10,20), (10,50), (10,70)]
        self.pool_sizes  :list[tuple] = [(2,2), (3,3)]
        self.pool_types  :list[str]   = ["avgpool", "maxpool"]
        self.act_types   :list[str]   = [None, 'sigmoid', 'relu', 'tanh']
        self.conv_layers :list[int]   = [1,2]
        self.lin_layers  :list[int]   = [0,1]
        self.dim         :list[int]   = [1,60,100] #Channel, Height, Width
        self.output_dim  :int         = 2 #Output Dimension
        self.filters     :list[int]   = [1,16,48,48] #Number of filters in the first layer

        self.generator2()
        self.save_net()

    def pooling(self, stride=1):
        new_list = []
        for start_str in self.layers:
            for p_type in self.pool_types:
                for size in self.pool_sizes:
                    padding = (size[0]-1, size[1]-1)
                    new_list.append(start_str + f'p; {p_type}; {size}; {stride}; {padding};; ')

    def linear(self, input_size:int):
        new_list = []
        kernel_size = (input_size, input_size)
        for start_str in self.layers:
                new_list.append(start_str + f'l; linear; {kernel_size};; ')
        self.layers = new_list

    def last_layer(self, input_size:int):
        new_list = []
        kernel_size = (input_size, self.output_dim)
        for start_str in self.layers:
                new_list.append(start_str + f'l; linear; {kernel_size};; ')
        self.layers = new_list

    def activation(self):
        new_list = []
        for start_str in self.layers:
            for activation in self.act_types:
                if activation is not None:
                    new_list.append(start_str + f'a; {activation};; ')
                else:
                    new_list.append(start_str)
        self.layers = new_list

    def conv(self, channel, stride: int = 1):
        new_list = []
        for start_str in self.layers:
            for kernel in self.conv_sizes:
                padding = (kernel[0] - 1, kernel[1] - 1)
                new_list.append(start_str + f'l; conv2d; {channel}; {kernel}; {stride}; {padding};; ')
        self.layers = new_list


    def generator(self):
        for num_conv in self.conv_layers:
            for i in range(num_conv):
                self.conv((self.filters[i], self.filters[i+1]))
                self.activation()
                self.pooling()
            self.layers = [i + 'v: view;;' for i in self.layers]
            for num_lin in self.lin_layers:
                for i in range(num_lin):
                    self.linear(self.filters[num_conv]*60*100)
                    self.activation()
                self.last_layer(self.filters[num_conv]*60*100)

    def generator2(self):
        for num_conv in self.conv_layers:
            for i in range(num_conv):
                self.conv((self.filters[i], self.filters[i + 1]))
                self.activation()
                self.pooling()
            # Nach den Conv-Layern wird ein View hinzugefügt
            self.layers = [i + 'v; view;; ' for i in self.layers]

            # Danach werden nur noch Linear-Layer hinzugefügt
            for num_lin in self.lin_layers:
                for i in range(num_lin):
                    self.linear(self.filters[num_conv] * 60 * 100)
                    self.activation()
                self.last_layer(self.filters[num_conv] * 60 * 100)

    def save_net(self):
        with open('netstruct.txt', 'w') as f:
            for layer in self.layers:
                f.write(layer + '\n')
        print('Netzstruktur gespeichert.')

Cnn = NetStruct()
