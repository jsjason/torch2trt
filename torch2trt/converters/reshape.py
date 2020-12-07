from torch2trt.torch2trt import *

@tensorrt_converter('torch.reshape')
def convert_layer_norm(ctx):
    input = get_arg(ctx, 'input', pos=0, default=None)
    shape = get_arg(ctx, 'shape', pos=1, default=None)
    output = ctx.method_return
    
    layer = ctx.network.add_shuffle(input._trt)
    layer.reshape_dims = tuple(shape)[1:] # (shape[0], shape[1], shape[2], shape[3])

    y_trt = layer.get_output(0)
    output._trt = y_trt

