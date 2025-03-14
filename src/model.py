import torch as th
import json as js

from torch.nn import (
    Conv2d,
    ConvTranspose2d,
    Sequential,
    BatchNorm2d,
    ReLU,
    Tanh,
    Module,
    Sigmoid,
    ModuleList,
    Linear,
    LayerNorm,
    Dropout,
    Upsample,
    Flatten,
    Softmax
)


_activations_ = {
    "relu": ReLU,
    "tanh": Tanh,
    "sigmoid": Sigmoid,
    "softmax": Softmax
}


class LinearBlock(Module):
    
    def __init__(
        self, 
        in_features: int,
        out_features: int,
        activation: str = "relu",
        dp: float = 0.0
    ) -> None:
        
        super().__init__()
        self._net_ = Sequential(
            Linear(in_features, out_features),
            LayerNorm(out_features),
            Dropout(dp),
            _activations_[activation]()
        )
    
    def __call__(self, inputs: th.Tensor) -> th.Tensor:
        return self._net_(inputs)
    
class MLP(Module):

    def __init__(self, params: dict) -> None:
        
        super().__init__()
        self.params = params
        self._layers_ = []
        layers_n = self.params["depth"]
        for idx in range(layers_n):

            layer = LinearBlock(**{
                param: value[idx]
                for (param, value) in self.params["params"].items()
            })
            self._layers_.append(layer)
        
        self._layers_ = ModuleList(self._layers_)
    
    def __call__(self, inputs: th.Tensor) -> th.Tensor:
        
        x = inputs
        for layer in self._layers_:
            x = layer(x)
        
        return x
    

class ConvBlock(Module):

    def __init__(
        self,
        in_channels: int, 
        out_channels: int,
        kernel_size: tuple = (2, 2), 
        padding: int = 0,
        stride: int = 2,
        activation: str = "relu",
        mode: str = "down"
    ) -> None:
        
        super().__init__()
        _conv_ = {
            "down": Conv2d(
                in_channels, out_channels, 
                kernel_size, stride, 
                padding
            ),
            "up": ConvTranspose2d(
                in_channels, out_channels, 
                kernel_size, stride, 
                padding
            )
        }
        self._net_ = Sequential(
            _conv_[mode],
            BatchNorm2d(num_features=out_channels),
            _activations_[activation]()
        )
    
    def __call__(self, inputs: th.Tensor) -> th.Tensor:
        return self._net_(inputs)


class DetectionHead(Module): 

    def __init__(self, n_classes: int, n_grids: int, params: dict) -> None:

        super().__init__()
        self.params = params
        
        self._mlp_ = MLP(self.params["mlp"])
        self._proj_ = LinearBlock(
            self.params["mlp"]["params"]["out_features"][-1], 
            self.params["patch_size"][0] * self.params["patch_size"][1] * 3
        )
        if "conv" in self.params:
            self._conv_ = []
            for idx in range(self.params["conv"]["depth"]):

                layer = ConvBlock(**{
                    param: values[idx]
                    for (param, values) in self.params["conv"]["params"].items()
                })
                self._conv_.append(layer)
            
            self._conv_ = ModuleList(self._conv_)

        self._out_ = [
            Upsample(size=n_grids),
            BatchNorm2d(num_features=(5 + n_classes)),
            Dropout(p=self.params["dp_rate"], inplace=True)
        ]
        if "conv" not in self.params:
            self._out_.insert(0, ConvBlock(
                self.params["in_channels"], 
                (5 + n_classes)
            ))
        
        self._out_ = Sequential(*self._out_)
        

    def __call__(self, inputs: th.Tensor) -> th.Tensor:

        x = inputs
        x = self._mlp_(x)
        x = self._proj_(x).view((inputs.size()[0], 3, ) + self.params["patch_size"])

        if "conv" in self.params:
            for layer in self._conv_:
                x = layer(x)

        return self._out_(x)
    
  
class Unet(Module):

    def __init__(self, params: dict) -> None:

        super().__init__()
        if isinstance(params, str):
            with open(params, "r") as params_f:
                params = js.load(params_f)

        self.down = []
        self.up = []

        for layer_idx in range(params["depth"]):

            self.down.append(ConvBlock(**{
                param: vals[layer_idx] 
                for (param, vals) in params["down"].items()
            }))
            self.up.append(ConvBlock(**{
                param: vals[layer_idx] 
                for (param, vals) in params["up"].items()
            }))
            
        
        self.down = ModuleList(self.down)
        self.up = ModuleList(self.up)
        self._act_ = _activations_["tanh"]()
    


    def __call__(self, inputs: th.Tensor) -> th.Tensor:

        down = inputs
        down_convs = []
        for layer in self.down:
            down = layer(down)
            down_convs.append(down)
        
        down_convs = down_convs[::-1][1:]
        up = down
        for i, layer in enumerate(self.up):
            up = layer(up)
            try:
                up = th.cat([up, down_convs[i]], dim=1)
            except:
                pass
        
        return up


            
class LossHead(Module):

    def __init__(self, n_classes: int, n_grids: int, params: dict) -> None:

        super().__init__()
        self.params = params
        mlp_conf = self.params["mlp"]
        grid_flt = (n_grids ** 2)
        self._flatten_ = Flatten(start_dim=1)

        self._bbox_head_ = Sequential(
            Linear(grid_flt * 4, mlp_conf["params"]["in_features"][0]),
            MLP(mlp_conf),
            Linear(mlp_conf["params"]["out_features"][-1], 4),
            _activations_[self.params["bbox_act"]]()
        )
        self._conf_head_ = Sequential(
            Linear(grid_flt, mlp_conf["params"]["in_features"][0]),
            MLP(mlp_conf),
            Linear(mlp_conf["params"]["out_features"][-1], 1),
            _activations_[self.params["conf_act"]]()
        )
        self._cls_head_ = Sequential(
            Linear((grid_flt * n_classes), mlp_conf["params"]["in_features"][0]),
            MLP(mlp_conf),
            Linear(mlp_conf["params"]["out_features"][-1], n_classes),
            _activations_[self.params["cls_act"]](dim=1)
        )



    def __call__(self, inputs: th.Tensor) -> th.Tensor:

        bbox_in = self._flatten_(inputs[:, :4, :, :])
        conf_in = self._flatten_(inputs[:, 4, :, :])
        cls_in = self._flatten_(inputs[:, 5:, :, :])

        return (
            self._bbox_head_(bbox_in),
            self._conf_head_(conf_in),
            self._cls_head_(cls_in)
        )

class DetectionNet(Module):


    def __init__(self, params: dict) -> None:

        super().__init__()
        self.params = params
        self._detection_head_ = DetectionHead(
            n_classes=self.params["n_classes"],
            n_grids=self.params["n_grids"],
            params=self.params["detection_head"]
        )
        self._loss_head_ = LossHead(
            n_classes=self.params["n_classes"],
            n_grids=self.params["n_grids"],
            params=self.params["loss_head"]
        )
    
    def __call__(self, inputs: th.Tensor) -> th.Tensor:

        detection_out = self._detection_head_(inputs)
        return self._loss_head_(detection_out)
        

if __name__ == "__main__":
    
    test = th.normal(0.12, 1.12, (1, 3, 512, 512))
    testA = th.normal(0.12, 1.12, (100, 32))

    model = DetectionNet({
        "n_grids": 12,
        "n_classes": 30,
        "detection_head": {
            "dp_rate": 0.2,
            "in_channels": 3,
            "patch_size": (128, 128),
            "mlp": {
                "depth": 3,
                "params": {
                    "in_features": [32, 64, 128],
                    "out_features": [64, 128, 215],
                    "dp": [0.45, 0.12, 0.0],
                    "activation": ["relu", "tanh", "relu"]
                }
            },
            "conv": {
                "depth": 3,
                "params": {
                    "in_channels": [3, 32, 64],
                    "out_channels": [32, 64, 35],
                    "mode": ["down", "down", "down"],
                    "kernel_size": [2, 2, 2],
                }   
            }
        },
        "loss_head": {
            "bbox_act": "relu",
            "conf_act": "sigmoid",
            "cls_act": "softmax",
            "mlp": {
                "depth": 3,
                "params": {
                    "in_features": [32, 64, 128],
                    "out_features": [64, 128, 215],
                    "dp": [0.45, 0.12, 0.0],
                    "activation": ["relu", "relu", "relu"]
                }
            }
        }
    })
    model_out = model(testA)
    print(model_out[0].size(), model_out[1].size(), model_out[2].size())




