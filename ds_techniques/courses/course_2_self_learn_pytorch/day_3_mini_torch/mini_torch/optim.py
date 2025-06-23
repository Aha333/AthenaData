# optim.py

class Optimizer:
    def __init__(self, parameters, lr=0.01):
        """
        参数:
            parameters: 可迭代对象，包含模型参数（Tensor）
            lr: 学习率
        """
        self.parameters = list(parameters)
        self.lr = lr

    def step(self):
        """
        执行一步参数更新。
        需要具体优化器子类实现。
        """
        raise NotImplementedError

    def zero_grad(self):
        """
        清空所有参数的梯度。
        """
        for param in self.parameters:
            param.zero_grad()

class SGD(Optimizer):
    def __init__(self, parameters, lr=0.01):
        super().__init__(parameters, lr)
        # 如果需要 momentum 逻辑，这里可以加，否则删除测试里的 momentum 参数

    def step(self):
        for param in self.parameters:
            if param.grad is not None:
                param.data -= self.lr * param.grad

class Adam(Optimizer):
    def __init__(self, parameters, lr=0.001, betas=(0.9, 0.999), eps=1e-8):
        super().__init__(parameters, lr)
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.m = [np.zeros_like(p.data) for p in self.parameters]  # 修正这里
        self.v = [np.zeros_like(p.data) for p in self.parameters]
        self.t = 0

    def step(self):
        self.t += 1
        for i, p in enumerate(self.parameters):
            if p.grad is None:
                continue
            grad = p.grad
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad ** 2)

            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)

            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

            
def train_step(model, optimizer, x, target):
    optimizer.zero_grad()
    out = model(x)
    loss = ((out - target) * (out - target)).sum()
    loss.backward()
    optimizer.step()
    return out.data.copy(), loss.data.copy(), {name: param.data.copy() for name, param in model._parameters.items()}

if __name__ == "__main__":
    import numpy as np
    from tensor import Tensor
    from nn import Linear

    np.random.seed(42)  # 固定随机种子，确保参数初始化相同

    x = Tensor([[1., 2., 3.]])
    target = Tensor([[0., 1.]])

    # 创建两个相同初始权重的模型（深拷贝参数）
    model_sgd = Linear(3, 2)
    model_adam = Linear(3, 2)

    # 同步参数（保证起点完全一致）
    for name in model_sgd._parameters:
        model_adam._parameters[name].data = model_sgd._parameters[name].data.copy()

    # 创建优化器
    optimizer_sgd = SGD(model_sgd.parameters(), lr=0.1)
    optimizer_adam = Adam(model_adam.parameters(), lr=0.1)

    # 训练一步
    out_sgd, loss_sgd, params_sgd = train_step(model_sgd, optimizer_sgd, x, target)
    out_adam, loss_adam, params_adam = train_step(model_adam, optimizer_adam, x, target)

    print("SGD Output:", out_sgd)
    print("SGD Loss:", loss_sgd)
    print("SGD Params:")
    for k, v in params_sgd.items():
        print(f"  {k}: {v}")

    print("\nAdam Output:", out_adam)
    print("Adam Loss:", loss_adam)
    print("Adam Params:")
    for k, v in params_adam.items():
        print(f"  {k}: {v}")