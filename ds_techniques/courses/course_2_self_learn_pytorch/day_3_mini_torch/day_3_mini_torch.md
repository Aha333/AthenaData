# 今天来看看pytorch的构成。 

context是在 学习构造自定义块；  - https://zh.d2l.ai/chapter_deep-learning-computation/model-construction.html ； 但是希望能更加系统的学习pytorch本身。 这里就用gpt和pytorch源代码学习了

`torch.nn` 是神经网络组建， 层， 容器。
`torch.optim` 是优化器
`torch.autograd 求导核心。 forward/backward机制
torch.utils 是数据加载， 模型保存 
torch.distributed 分布训练框架
torch.tensor.py 是tensor类型封装。 

更新： 
下面这个总结非常好 - 
功能	代码位置示例	说明
模型结构和参数管理	Module 及其子类（Linear、Conv等）	负责定义层的组合，保存权重
具体计算细节	Tensor 类及其操作（加法、乘法、matmul）	实现数学运算和自动求导
优化算法	Optimizer 类（如 SGD、Adam）	使用梯度调整模型参数
损失函数	Loss 函数（MSELoss、CrossEntropy）	计算训练目标

Module 是模型结构和参数的“组织者”。不管算法。 
我们说算法是说的： 具体算法是底层数学运算、自动求导和优化更新规则。

一个例子
Module 定义了一个 Linear 层，保存了权重 weight 和偏置 bias。
Tensor 定义了加法、乘法、矩阵乘法的运算和自动求导规则。
训练时，Module 调用 Tensor 的运算完成前向传播，计算出输出。
计算损失后，调用 Tensor.backward() 计算梯度。
优化器根据梯度更新 Module 中的参数。



# 最好的办法是写一个mini torch。

# 第一节 Tensor
第一步 先测试一下tensor。 tensor不光是一个数据
```
cd ~/Documents/AthenaData/ds_techniques/courses/course_2_self_learn_pytorch/day_3_mini_torch
python mini_torch/tensor.py
```
结果如下
```
if __name__ == "__main__":
    a = Tensor(2.0, requires_grad=True)
    b = Tensor(3.0, requires_grad=True)
    c = a * b + a + b  # 构建前向图
    c.backward()       # 反向传播
    print(a)  # Tensor(data=2.0, grad=4.0)
    print(b)  # Tensor(data=3.0, grad=3.0)
    print(c)  # Tensor(data=11.0, grad=1)
```
问题1: 这里a不是一个标量吗 为什么会有导数grad呢？
首先， 一开始的grad就是0 本身。 只有在backward以后才会更新grad！
第二， 要明白这里导数的意义。 这里导数的是说的每一个输入节点a,b的变化 对输出c的 变化。这就是为什么是c.backward（）了。 表示指定了这里的output变量。

所以 - 标量 Tensor 依然可以有梯度； grad 保存的是“该 Tensor 对最终标量输出的导数值”。


问题2： “_backward() 是定义在一个内部函数里，
它引用了 out.grad，但是这个 out 并没有存储到 Tensor 里，那它怎么还能访问到呢？”
答案是 python的机制 函数内部定义另一个函数，并且这个内部函数用到了外部函数的局部变量，
那么 Python 会把这些局部变量“打包”进这个函数对象的闭包中

如果你不用闭包，还能做吗？PyTorch 的 C++ 内核（autograd.Function）里的常用方式，因为它比闭包显式、可控、性能可预测。
```
def __mul__(self, other):
    ....
    out._saved_self = self
    out._saved_other = other
    def _backward():
        out._saved_self.grad += out._saved_other.data * out.grad
```

!!!当我们定义 _backward() 时，闭包里保存的是对那个 out Tensor 对象 的引用（指针），
不是它 .grad 属性的拷贝。因此，只要你修改了 out.grad，闭包里的 out 引用看到的是同一个对象，当然这个 .grad 也变了。
```
out = Tensor(3.0)
def foo():
    print(out.grad)

out.grad = 5
foo()  # 会打印 5，因为闭包里的 out 指向的是同一个对象
```

问题3
c = a*b, 那么grad c to a = b. 
为什么反向传播代码里_backward ()中不是b.data 而是b.data * c.grad？？
不理解了

# 两个backward
* _backward() 
- 作用：它对应的是这个操作本身的·反向传播规则·。
- 由谁定义	： 具体的操作函数（+、*、sin等）
- 触发时机： 每个由操作生成的 Tensor 都会被附带一个 _backward() 函数。
- 只做一步局部链式法则；不管整个图，只知道“自己的输入输出关系”；由操作定义，随 Tensor 一起保存


* backward()
- 全图反向传播触发器
- 从“最终输出 Tensor”出发， 构建计算图的拓扑排序列表；按照拓扑顺序倒序执行所有节点的 _backward()；逐步累积和计算所有中间节点和输入节点的梯度
- 调用所有节点的 _backward()，连接链式法则；是自动微分的“主控流程”



# 面试题： 
## 1 实现一个简化版的 Tensor 类，支持加法和乘法运算，并且能够对标量结果执行反向传播，计算梯度。
## 题目3：解释你在实现 backward() 时，为什么需要先做拓扑排序？
##

更新： 
1. ！！这里加入了tensor的非标量运算sum/add， 涉及到广播问题。 我们之后在来解决。
2. 只要给 Tensor 类加一个魔法方法 __matmul__：， 
def __matmul__(self, other):
    return self.matmul(other)
c = a @ b  # 自动调用 a.__matmul__(b)
这里@是 Python 的魔法方法（special method），专门用来重载 @ 运算符的。

# 第二节 设计nn模块

- 实现一个 Module 基类 

Module就是为了管理模型结构的， 不是为了具体的计算参数！

我们来看看module类的设计初衷， 解决了什么问题 - 在构建复杂网络时，问题自然出现： 参数不好管理

最后 Module 类设计总结

- **要解决的问题：管理模型参数和模块层级结构**  
  - **解决方案：** 使用两个字典 `_parameters` 和 `_modules` 分别保存参数和子模块，支持递归管理和访问。

- **`__init__`**  
  - 初始化两个空字典 `_parameters` 和 `_modules`，为后续参数和子模块存储做准备。

- **`__setattr__`**  
  - 重写属性赋值行为。  
  - 当赋值的是 `Tensor` 实例时，自动把它注册到 `_parameters` 字典中（方便统一管理训练参数）。  
  - 当赋值的是 `Module` 实例时，自动把它注册到 `_modules` 字典中（方便管理嵌套子模块）。  
  - 其他属性正常赋值。

- **`parameters`**  
  - 用来递归生成当前模块和所有子模块的参数。  
  - 方便训练时统一拿到所有需要优化的参数。

- **`forward`**  
  - 抽象方法，子类必须实现。定义模块的前向计算逻辑。

- **`__call__`**  
  - 使得模块实例本身可调用。  
  - 调用时自动执行 `forward`，简化调用方式。- >  支持了 model(x) 这样的写法， 不需要module。forward(x)

！！以后再研究 -这里程序写法很精妙， 比如为什么调用super。__setattr__。 
问题：这里用来递归生成当前模块和所有子模块的参数的时候， 如何保证每次顺序一样呢？我看到_modules 和_parameters 都是set不是list
答案： _modules 和 _parameters 实际上在代码里是字典（dict），不是 set。Python 的字典在 Python 3.7 及以后的版本中，是保持插入顺序的，也就是说，遍历字典时，键值对的顺序是你添加的顺序。
* 如果你用的是 Python 3.6 之前的版本，字典顺序不保证，但现在大多数环境都用的是 3.7+，不太需要担心这个问题。
* 如果需要严格控制顺序，也可以改成用 collections.OrderedDict 来保证（不过现在没必要）。


# 第三节 ReLU, MSELoss, Sequential 组件。
* 问题： 为什么损失函数（比如MSELoss）也用Module来实现，而它本质上是个操作而不是“模块”呢？

损失函数放这里是为了统一接口、方便管理和复用，即使它不含参数，也方便你像调用层一样调用它


* 问题1： 为什么PyTorch源码里的 mse_loss 函数没有显示定义 backward 方法。
这是因为PyTorch的自动微分系统（Autograd）是基于计算图和C++内核实现的，具体原理是：
mse_loss 函数调用了底层C++扩展接口（torch._C._nn.mse_loss），这是一个内置操作，内部已经实现了前向计算和反向传播逻辑。

如果自己用纯Python/NumPy写自动微分，就需要显式写出backward函数，比如我们之前写的 Tensor 类里的 _backward。

对比
```
class MSELoss(Module):
    def forward(self, pred, target):
        diff = pred - target
        out = Tensor(np.mean(diff.data ** 2), requires_grad=pred.requires_grad or target.requires_grad)

        def _backward():
            if out.grad is None:
                return
            grad = (2 / diff.data.size) * diff.data * out.grad
            if pred.grad is None:
                pred.grad = np.zeros_like(pred.data)
            if target.grad is None:
                target.grad = np.zeros_like(target.data)
            pred.grad += grad
            target.grad -= grad  # target是常量时一般不更新，这里算梯度方便

        out._backward = _backward
        out._prev = {pred, target}
        return out
vs。
class MSELoss(Module):
    def __init__(self):
        super().__init__()

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        # 计算均方误差：(input - target)^2 的均值
        diff = input - target
        squared = diff * diff
        loss = squared.sum() * (1.0 / diff.data.size)  # 平均值
        return loss
```

* 问题： 算子在哪一个class里面实现？？——Tensor 还是 Module？ 因为tensor需要实现_backward(), 然后module需要实现forward。这里怎么划分的呢？

其实是深度学习框架设计里的一个经典问题，咱们可以这样区分：
- 基础张量操作放 Tensor，负责数据和梯度传播
- 算子是实现“前向计算”的代码，既可以是 Tensor 的方法，也可以是 Module 的 forward 逻辑调用 Tensor 操作
- 损失函数放 Module 是为了统一模块调用接口，方便组合和管理，但核心计算（均方差、交叉熵）其实还是基于 Tensor 的操作

* 自动求导的原理
注意， 两个tensor的`基本操作`， 得到一个新的tensor， 这里的backward自动生成的前提是， 这个 `基本操作`已经被定义了backward的函数。 这样得到的tensor会自动asssign一个backward方法。非常巧妙！ 如果这个操作没有定义， 那么就需要重写backwar了。 

module当做一个算子：输入tensor， 得到另一个tensor， 由于是module的合集
* 理解： module类的forward方法相当于一个tensor算子
- 底层算子： Tensor.__add__， 比如a+b
- 中间级别算子： ReLU, MSELoss， 比如ReLU(x), loss(pred, target)
- 网络层： Module.forward()， 比如Linear(x)

* 也就是说： nn.Module 负责把计算组织成有结构、有状态、可组合的对象。是否含参数，不是关键。


更进一步理解
*  Sequential是一个take 1 个Tensor的算子
*  MSELoss 是一特take 2个Tensor 的算子， 输出的是一个标量。 虽然MSELoss输出是标量， 但是是一个Tensor 对象。 这个设计是为了统一计算图和自动求导系统。 了所有参与计算的值都封装成了 Tensor 对象


## 第三节补充
这里开始进入高级部分。 train和eval本身都是比较高级的topic。
目的是“模型的训练逻辑和模型的行为逻辑解耦。”
- 有些层在训练和测试阶段是行为不同的
- 比如 Dropout 和 BatchNorm
训练时（.train()）：启用 Dropout、BatchNorm 动态统计
评估时（.eval()）：禁用 Dropout，使用固定的均值方差
问题你会问了： 为什么不直接写两套 forward？
-> 这是高度封装 + 最小重复的设计思路。 


理论问题： 所以为什么会出现train 和eval不同的模式？？
第一： Dropout 是怎么想出来的 可以证明收敛并且得到的参数无偏吗
： 参数无偏性 - 在数学期望意义下，Dropout 产生的梯度是无偏的。

实现code
并不难：
* 在module clasa里面， 加上 self.training状态=True/False而已； 然后要递归每一个sub module的training 状态变量。
* 具体的用法是主要就是影响模型前向传播（forward）的行为。
train() 和 eval() 不直接影响 backward 计算，只影响 forward 的行为。
* 只有包含 特定层（如 Dropout、BatchNorm） 的模块才会真正“用上”这个模式切换
普通层（Linear, Conv, ReLU, Sequential等）本身没有模式相关行为，它们的 forward() 不受 train() / eval() 影响。

## Dropout module implement 的时候的 _backward 是怎么回事？Dropout 操作怎么会有梯度呢？
首先， 这里是一个tensor to tensor 的算子， 而别不是普通算子。 所以需要重新写_backward。
可是这个梯度的意思是什么么？其实可以完全推理的。 就是简单的根据偏导数而已。 因为数学上， drop out是对每一个x的元素乘以了一个 0/1 然后缩放了p。 这个0/1是抽样波努力生成的0/1. 
在mask已经生成（0/1已经生成）的情况下， 其实就是一个线性变换了。 对应的变换矩阵可以当作一个diag(mask)

## batchnorm
原理： BatchNorm 本身不是为了解决 mini-batch “代表性不足”的问题而设计的，而是解决激活分布变化（Internal Covariate Shift）。： 当神经网络中某一层的参数发生更新后，其输出的分布就会改变
* 这里说的“分布”指的是每个特征（feature）在当前 batch 上的边缘分布，而不是所有特征的联合分布。BatchNorm 是对每个维度（feature/channel）单独计算均值和方差，然后归一化。没有直接考虑不同特征之间的联合分布或协方差矩阵，只是对每个 feature 独立归一化

Input → Layer1 → BatchNorm → Activation → Layer2

问题： 为什么要放在激活函数之前？
这是标准做法，原因如下：

BatchNorm 归一化的是 线性层的输出，也就是激活函数的输入。

如果你在激活之后再归一化，可能会破坏激活函数的非线性特性，尤其像 ReLU 这种对负数“裁剪”的激活。
问题： batchnorm的梯度怎么算的
BatchNorm 的梯度确实比普通层复杂不少！ 不像dropout比较简单


 


BatchNorm 的反向传播推导



为什么它对小 batch 不稳定？

LayerNorm、InstanceNorm、GroupNorm 对比

用 BatchNorm 替代 Bias？

BatchNorm + Dropout 会冲突吗？


** 之前老板说了 regression下面不能 drop out来着。 是什么意思 貌似和SGD算法有关？
如果你使用 BatchNorm（批量归一化）或其他正则化技术，Dropout 可能效果有限，甚至干扰训练。？？？


## 理论： 非可学习参数 vs 可学习参数
Dropout 的概率 p 虽然影响模型行为，但它本身是一个固定超参数，不参与梯度计算，也不被训练优化，因此属于非可学习参数。


# 第四节 optim.py
作用： 参数更新逻辑解耦
* Optimizer基类
* 具体优化器实现 (如 SGD, Adam)

有意思的是： 完全对！优化器本质上只关心「参数们」本身，而不是模型的结构或细节。
* 问题： optim的借口只take了module的param。 难道它不需要知道这些参数属于哪个 Module，也不关心这些参数是怎么组合成模型吗？？？
不需要模型结构本身！！！
一旦得到梯度，优化器只关心“参数 + 梯度”这对信息，不需要知道模型拓扑或层次。

## 训练标配 - 三步

for data, target in dataloader:
    optimizer.zero_grad()   # 1. 清零梯度
    output = model(data)    # 前向
    loss = loss_fn(output, target)
    loss.backward()         # 2. 计算梯度
    optimizer.step()        # 3. 更新参数


到此为止 我们实现了基本功能了。 

# 第五节 utils/DataLoader
动机： 难道Tensor本身不就是Data吗？为什么有这么一个类

Tensor 只能存储数字数组，而数据集通常是成千上万个样本，且每个样本是独立的单位。
训练时常常需要按批处理数据，Tensor 不负责分批操作。
假设你有一个包含10万张图片的数据集，保存在硬盘：Tensor 方案：你要么一次性把10万张图片全部加载成一个大Tensor（内存炸了），要么没法用Tensor直接存储。Dataset + DataLoader 方案：Dataset 告诉你第 i 张图片在哪里，怎么读取和预处理它
DataLoader 负责每次从 Dataset 取出 batch_size 张图片，预处理成Tensor，再送给模型。

所以这里面有两个逻辑
1. 定义dataset
2. 定义datloader： 如何把数据批量加载。 包括这么几个课题
- 批处理
- 打乱顺序
- 并行加载
- 多线程
- 自动拼接
- 支持各种采样策略
等等

## Dataset的核心接口
- __len__() ： 不是拿数据的，它告诉框架：“我总共有多少个样本”。 
- __getitem__() ： 定义“怎么访问第 i 个样本” 
问题： 为什么 __len__ 很重要
比如 - DataLoader 会先生成 index 列表。 
indices = list(range(len(dataset)))  # 如果 shuffle=True，还会 random.shuffle(indices)

## 而 DataLoader ： 它不是操作“数据”本身，而是操作“索引”！
* 它不会一次性加载整个数据（那样太浪费内存）： 
* 它只需要知道：你有多少条样本（__len__）； 
* 然后按顺序或打乱后，每次抽出一批“索引”去 dataset[i]


# 第六节 实现 Module.train() 和 Module.eval() 方法





----- appendix

# 关于autograd vs module的关系（略）

nn.Module = 神经网络的结构 + 参数容器（像一个工厂）

autograd.Function = 自定义操作符的微分规则（像工厂里的机器）

两者相辅相成，Module 组织结构，Function 提供算子。绝大部分用户不需要写 Function，写 Module 就够了。




5. 扩展特性
实现 DataLoader 来方便批量加载数据。
支持更多层和激活函数。
实现设备（CPU/GPU）抽象（如果你想玩更高级的）。

6. （可选）性能调优和 profiler
简单实现性能监测，了解瓶颈

但这属于进阶部分





完善 Module 的训练流程

实现 Module.train() 和 Module.eval() 方法，控制训练/推理模式（比如 dropout，batchnorm 在训练和推理阶段行为不同）

这样可以为后续扩展 BatchNorm、Dropout 等模块打基础。

实现更多基础层和激活函数

比如：ReLU, Sigmoid, Tanh 等激活函数模块

卷积层 Conv2d（如果准备做卷积神经网络）

Dropout 层（配合上面的训练/推理模式）

完善 Loss 体系

实现更多损失函数，比如交叉熵 CrossEntropyLoss

支持分类问题和多类别问题

训练脚本封装与实验

实现完整的训练循环，包括：

数据加载（你可以写简单的 DataLoader）

模型前向+反向传播+优化器步骤

训练日志输出、评估指标计算

这样可以验证优化器、模块、tensor 的整体联动

实现自动求导引擎优化

目前自动求导还很基础，可以逐步完善计算图，支持更复杂的操作

支持更多 Tensor 运算符

实现序列模型和高级模型架构

比如 RNN、Transformer 等，基于你已有的框架搭建

