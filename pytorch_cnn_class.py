class Net(nn.Module):
    def __init__(self, num_classes, channels, 
                 conv_kernels, conv_strides, conv_paddings, 
                 pooling_kernels, pooling_strides, pooling_paddings,
                 linears, 
                 dropouts,
                 input_dim):
        super(Net, self).__init__()
        
        feature_layers = []
        classification_layers = []

        #Features Layers
        for index in range(1, len(channels)):
            feature_layers.append(nn.Conv2d(in_channels=channels[index-1],
                                  out_channels=channels[index], 
                                  kernel_size=conv_kernels[index-1], 
                                  stride=conv_strides[index-1],
                                  padding=conv_paddings[index-1]))

            feature_layers.append(nn.ReLU())

            if pooling_kernels[index-1] != 0:
                feature_layers.append(nn.MaxPool2d(pooling_kerenls[index-1], 
                                           pooling_strides[index-1], 
                                           pooling_paddings[index-1]))
                
        self.features = nn.Sequential(*feature_layers)
        
        #Classification Layers
        self.num_cnn_output = functools.reduce(operator.mul, list(self.features(torch.rand(1, *input_dim)).shape))

        classification_layers.append(nn.Linear(self.num_cnn_output, linears[0]))
        classification_layers.append(nn.ReLU())

        if dropouts[0] != 0:
              classification_layers.append(nn.Dropout(p=dropouts[0]))

        for index in range(1, len(linears)):
            classification_layers.append(nn.Linear(linears[index-1], linears[index]))
            classification_layers.append(nn.ReLU())
            if dropouts[index] != 0:
              classification_layers.append(nn.Dropout(p=dropouts[index]))

        self.classification = nn.Sequential(*classification_layers)
    
    def forward(self, x):
        output = self.features(x)

        #flatten the output
        output = output.view(-1, self.num_cnn_output)

        output = self.classification(output)

        return output

num_classes = 50

channels = [3,64,192,384,256,256]
conv_kernels = [11,5,3,3,3]
conv_strides = [4,1,1,1,1]
conv_paddings = [2,2,1,1,1]

pooling_kerenls = [3,3,0,0,3]
pooling_strides = [2,2,0,0,2]
pooling_paddings = [0,0,0,0,0]

linears = [4096,4096,1000]

dropouts = [0.5, 0, 0]

input_dim = images[0].shape

model_scratch = Net(num_classes, channels, 
                    conv_kernels, conv_strides, conv_paddings, 
                    pooling_kerenls, pooling_strides, pooling_paddings,
                    linears, 
                    dropouts,
                    input_dim)
