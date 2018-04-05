def init_weights(self):
    # If the initial values of the coefficients are provided,
    # flip the coefficients respectively for easier computation.
    # Initialize by module types
    for name, param in self.named_parameters():
        if name == 'a_linear.weight':
            param.data = torch.FloatTensor(self.a_kernel)
        elif name == 'b_conv.weight':
            param.data = torch.FloatTensor(self.b_kernel)

def __init__(self,b_kernel_size,a_kernel_size,b_kernel=None,a_kernel=None):
    super(LFilter,self).__init__()
    assert b_kernel is None or (not(b_kernel is None) and b_kernel_size==len(b_kernel)), 'b_kernel does not match to b_kernel_size.'
    assert a_kernel is None or (not(a_kernel is None) and a_kernel_size==len(a_kernel)), 'a_kernel does not match to a_kernel_size.'
    self.b_conv = nn.Conv1d(1,1,kernel_size=b_kernel_size,padding=b_kernel_size-1,bias=False)
    #self.a_linear = nn.Conv1d(1,1,kernel_size=a_kernel_size, padding=a_kernel_size)
    self.a_linear = nn.Linear(a_kernel_size-1,1,bias=False)
    if b_kernel is not None:
        self.b_kernel = torch.FloatTensor(np.flip(b_kernel,axis=0).copy()).view(1,1,-1)
    if a_kernel is not None:
        self.a_kernel = torch.FloatTensor(np.flip(np.divide(a_kernel[1:],a_kernel[0]+0.0),axis=0).copy()).view(1,-1)
    self.a_kernel_size = a_kernel_size
    self.b_kernel_size = b_kernel_size
    self.init_weights()
