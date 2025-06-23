# nn.py

from tensor import Tensor  # 假设 tensor.py 在同目录
import numpy as np

class Module:
    def __init__(self):
        self._modules = {}
        self._parameters = {}
        self.training = True  # 默认是训练模式

    def parameters(self):
        for param in self._parameters.values():
            yield param
        for module in self._modules.values():
            yield from module.parameters()

    def __setattr__(self, name, value):
        if isinstance(value, Tensor):
            self._parameters[name] = value
        elif isinstance(value, Module):
            self._modules[name] = value
        super().__setattr__(name, value)

    def train(self, mode=True):
        """
        Sets the module in training mode.
        
        Args:
            mode (bool): whether to set training mode (True) or evaluation mode (False)
        """
        self.training = mode
        for module in self._modules.values():
            module.train(mode)
        return self

    def eval(self):
        """
        Sets the module in evaluation mode.
        """
        return self.train(False)

    def forward(self, *inputs):
        raise NotImplementedError

    def __call__(self, *inputs):
        return self.forward(*inputs)


class Linear(Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = Tensor.randn(out_features, in_features)
        self.bias = Tensor.randn(out_features)

    def forward(self, x):
        return x.matmul(self.weight.T()) + self.bias


class ReLU(Module):
    def forward(self, x):
        out = Tensor((x.data > 0) * x.data, requires_grad=x.requires_grad)

        def _backward():
            if out.grad is None:
                return
            if x.grad is None:
                x.grad = np.zeros_like(x.data)
            x.grad += (x.data > 0) * out.grad

        out._backward = _backward
        out._prev = {x}
        return out


class Dropout(Module):
    """
    Dropout module that behaves differently in train vs eval mode.
    
    In training mode: randomly sets elements to zero with probability p
    In evaluation mode: scales the output by (1-p) to maintain expected value
    """
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
        self.scale = 1.0 / (1.0 - p)  # 缩放因子，用于推理时

    def forward(self, x):
        if self.training:
            # 训练模式：随机丢弃
            mask = np.random.binomial(1, 1-self.p, size=x.data.shape)
            out = Tensor(x.data * mask * self.scale, requires_grad=x.requires_grad)
            
            def _backward():
                if out.grad is None:
                    return
                if x.grad is None:
                    x.grad = np.zeros_like(x.data)
                x.grad += out.grad * mask * self.scale
            
            out._backward = _backward
            out._prev = {x}
            return out
        else:
            # 评估模式：直接返回输入（已经通过scale调整了期望值）
            return x


class MSELoss(Module):
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        diff = input - target
        out = Tensor((diff.data ** 2).mean(), requires_grad=input.requires_grad or target.requires_grad)

        def _backward():
            if out.grad is None:
                return
            grad = (2 / diff.data.size) * diff.data * out.grad
            if input.grad is None:
                input.grad = np.zeros_like(input.data)
            if target.grad is None:
                target.grad = np.zeros_like(target.data)
            input.grad += grad
            target.grad -= grad  # 方便调试

        out._backward = _backward
        out._prev = {input, target}
        return out


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        for idx, module in enumerate(modules):
            self._modules[str(idx)] = module
            setattr(self, str(idx), module)

    def forward(self, x):
        for module in self._modules.values():
            x = module(x)
        return x


if __name__ == "__main__":
    import numpy as np

    # 构建包含 Dropout 的模型：Linear -> ReLU -> Dropout -> Linear
    model = Sequential(
        Linear(3, 4),
        ReLU(),
        Dropout(p=0.5),  # 50% 的 dropout
        Linear(4, 2)
    )

    # 测试数据
    x = Tensor([[1.0, -1.0, 2.0]], requires_grad=True)

    print("=" * 50)
    print("Testing Dropout with train/eval modes")
    print("=" * 50)

    # 训练模式测试
    print("Training mode:")
    model.train()
    print(f"Model training mode: {model.training}")
    
    # 多次前向传播，观察 Dropout 的随机性
    for i in range(3):
        out_train = model(x)
        print(f"  Forward {i+1}: {out_train.data}")

    # 评估模式测试
    print("\nEvaluation mode:")
    model.eval()
    print(f"Model training mode: {model.training}")
    
    # 多次前向传播，观察输出的一致性
    for i in range(3):
        out_eval = model(x)
        print(f"  Forward {i+1}: {out_eval.data}")

    print("\n" + "=" * 50)
    print("Notice: In training mode, outputs vary due to dropout randomness.")
    print("        In evaluation mode, outputs are consistent.")
    print("=" * 50)
