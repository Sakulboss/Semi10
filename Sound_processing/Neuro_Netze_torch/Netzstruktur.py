#l; conv; (1,16); (3,3); 1; 1;; a; relu;; p; maxpool; (3,3); 1; 1;; l; conv; (16, 32); (3,3); 1; 1;; v; view;; l; linear; (204800,5);;

class NetStruct:
    def __init__(self):
        self.layers = ""
        self.conv_sizes = [(1,1), (2,2), (3,3), (10,20), (10,50), (10,70)] #[(1,1), (2,2), (3,3), (4,4), (5,5), (6,6), (7,7), (8,8), (9,9), (10,20), (10,50), (10,70)]
        #self.conv_layers = [1,2,3]
        self.pool_conv = ['', 'sigmoid', 'relu', 'tanh']
        self.pool_linear = ['', 'sigmoid', 'relu', 'tanh']
        self.linear_layer = [1,2,3]

    @staticmethod
    def pooling(p_type:str, size, stride, padding):
        return f'p; {p_type}; {size}; {stride}; {padding};;'

    @staticmethod
    def linear_layer(kernel_size:tuple):
        return f'l; linear; {kernel_size};;'

    @staticmethod
    def activation_layer(activation:str):
        return f'a; {activation};;'

    @staticmethod
    def conv_layer(channel: tuple, kernel_size: tuple, padding: int, stride: int = 1):
        return f'l; conv2d; {channel}, {kernel_size}; {stride}; {padding};;'

    def generator(self):
        liste = []
        for conv_size in self.conv_sizes:
            for pool_conv in self.pool_conv:
                for linear_layer in self.linear_layer:
                    for pool_linear in self.pool_linear:
                        liste.append()

