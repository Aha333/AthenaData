import numpy as np

class Tensor:
    def __init__(self, data, requires_grad=False):
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        self.data = data
        self.requires_grad = requires_grad
        self.grad = None
        self._backward = lambda: None
        self._prev = set()

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, requires_grad=self.requires_grad or other.requires_grad)

        def _backward():
            if out.grad is None:
                return

            grad = out.grad

            if self.requires_grad:
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                self.grad += grad if self.data.shape == grad.shape else grad.sum(axis=0)

            if other.requires_grad:
                if other.grad is None:
                    other.grad = np.zeros_like(other.data)

                grad_other = grad
                # 自动广播回传
                while grad_other.ndim > other.data.ndim:
                    grad_other = grad_other.sum(axis=0)
                for i in range(other.data.ndim):
                    if other.data.shape[i] == 1 and grad_other.shape[i] > 1:
                        grad_other = grad_other.sum(axis=i, keepdims=True)

                other.grad += grad_other

        out._backward = _backward
        out._prev = {self, other}
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, requires_grad=self.requires_grad or other.requires_grad)

        def _backward():
            if out.grad is None:
                return
            if self.requires_grad:
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                self.grad += other.data * out.grad
            if other.requires_grad:
                if other.grad is None:
                    other.grad = np.zeros_like(other.data)
                other.grad += self.data * out.grad

        out._backward = _backward
        out._prev = {self, other}
        return out
    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)
    
    def matmul(self, other):
        out = Tensor(self.data @ other.data, requires_grad=self.requires_grad or other.requires_grad)

        def _backward():
            if out.grad is None:
                return
            grad = out.grad
            if self.grad is None:
                self.grad = np.zeros_like(self.data)
            if other.grad is None:
                other.grad = np.zeros_like(other.data)

            self.grad += grad @ other.data.T
            other.grad += self.data.T @ grad

        out._backward = _backward
        out._prev = {self, other}
        return out

    def T(self):
        out = Tensor(self.data.T, requires_grad=self.requires_grad)

        def _backward():
            if out.grad is None:
                return
            if self.grad is None:
                self.grad = np.zeros_like(self.data)
            self.grad += out.grad.T

        out._backward = _backward
        out._prev = {self}
        return out

    def sum(self):
        out = Tensor(self.data.sum(), requires_grad=self.requires_grad)

        def _backward():
            if out.grad is None:
                return
            if self.grad is None:
                self.grad = np.zeros_like(self.data)
            self.grad += out.grad * np.ones_like(self.data)

        out._backward = _backward
        out._prev = {self}
        return out

    def backward(self):
        topo = []
        visited = set()

        def build_topo(t):
            if t not in visited:
                visited.add(t)
                for child in t._prev:
                    build_topo(child)
                topo.append(t)

        build_topo(self)

        if self.grad is None:
            self.grad = np.ones_like(self.data)

        for t in reversed(topo):
            t._backward()

    def zero_grad(self):
        self.grad = None

    @staticmethod
    def randn(*shape):
        data = np.random.randn(*shape)
        return Tensor(data, requires_grad=True)

    def __repr__(self):
        return f"Tensor(data={self.data}, grad={self.grad})"


# 示例测试：标量 backward
if __name__ == "__main__":
    a = Tensor(2.0, requires_grad=True)
    b = Tensor(3.0, requires_grad=True)
    c = a * b + a + b  # forward
    c.backward()       # backward

    print('----- 第一次 backward -----')
    print(a)  # grad = 4
    print(b)  # grad = 3
    print(c)  # grad = 1

    # zero_grad + 再次反向传播
    a.zero_grad()
    b.zero_grad()
    c = a * b + a + b
    c.backward()

    print('----- 第二次 backward -----')
    print(a)
    print(b)
    print(c)
