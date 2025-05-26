
# 原理
Logits 是神经网络输出的原始值（通常是在最后一层全连接层后得到的）。这些 logits 没有经过任何归一化处理，因此它们可以是任意实数。它们并不代表概率，而是每个类别的“评分”或“相对优势”。
虽然 Softmax 将 logits 转换为概率，但 优化的真正对象是 logits，而非概率。通过调整 logits，使得模型最终的预测概率越来越接近真实标签的概率。

# 常见面试题
1) Softmax 函数是什么吗 
Softmax 是一个将任意实数向量转换为概率分布的函数，通常用于多分类问题的输出层。
2) logit 和softmax关系 
logits 实际上就是 log-odds。它们是 Softmax 输出概率之间的对数几率.Softmax 是一个将 logits 转换为 概率分布 的函数
3) 为什么指数性的softmax形式
Softmax函数的核心是通过指数函数来“放大”输入的差异；指数函数是连续且可微的；保证输出的所有值都为正数来归一化
4) Sigmoid 和 Softmax 的区别和联系
Sigmoid 输入是单一变量，通常用于二分类或多标签分类问题，输出一个概率值。
Softmax 输入是向量. 当你有多个类别，并且每个输入只能属于一个类别时，Softmax是一个理想的选择。
5) softmax vs. cross-entropy
在神经网络中，通常会将模型的原始输出（logits）通过softmax函数转换成概率分布，再计算Cross-Entropy:交叉熵损失是专门设计来衡量概率模型的输出和真实标签之间的差异。

6) softmax vs. cross-entropy derivative
重新理解一下关于loss function. 
cross-entropy 首先是概率模型下的loss function。
 每一个row， 虽然是 k个分类的 sum of y_i*log(p(y_i)|x), 但是注意， 实际只有一个realized y_i==1,    其余的y_i==0. 

所以最后其实是 - log(p(y_i)) 而已. 

第二： softmax的梯度求导（对于logic求导）性质非常特别。正确类别， 梯度是p(y_i)-1, 错误就是 p(y_i)本身（这里p（y_i）就是softmax）. Indeed, the derivative of the softmax itself requires some computation, but the combination with the cross-entropy loss does simplify the gradients quite nicely.
6) 反向传播的高效计算
对于softmax 对lgits求导， 结果是一个jacobi矩阵。 因为softmax输出是向量， 再对n维度输入求导。这里已经比较简化了。
基本上结果element是 p(y)(1-p(y))或者 p(yi)p(yj).

对于entropy loss， 是一个标量， 导数就是一个向量而已。 而且都不需要“计算”了。 就是 P(y) -1{y=i}. 这避免了复杂的雅可比矩阵乘法，因为不再需要显式地计算每个类别之间的相互影响。
例子
Logits: 
[2.0,1.0,0.1]
Softmax 输出: 
[0.6652,0.2447,0.0901]（假设 Softmax 已经计算过了）
给定真实标签
y=[0,1,0]，
那么
partial(L/z) = [p(y1), p(y2)-0, p(y3)] = [0.6652,0.2447-1,0.0901] 
7) softmax 数值稳定性 -Log-Sum-Exp Trick
常见的技巧是减去输入向量的最大值。 torch.nn.functional.softmax 来实现的，PyTorch 会自动处理数值稳定性的问题。
8) softmax的复杂度
一共三层计算 每一个logit z 算 e(z_i). 必须每一个类别都算， 这就是O(K), K 是类别。 
然后再归一化。 还是O(K)
9) when K is super large. 比如K是所有的词汇量。 有哪些技巧
比较高级的工程问题了

是的，Word2Vec 的目标函数并不是直接的 交叉熵（cross-entropy）损失函数。尽管它看起来和交叉熵损失相似，但在实际实现中，Word2Vec 采用的是一种 特殊的优化方法，尤其是在 负采样（Negative Sampling）和 层次 Softmax（Hierarchical Softmax）两种变种中。
* 在传统的 Softmax 中，模型通过 Softmax 函数 计算每个上下文词和目标词之间的概率。但由于词汇量 V 非常大，计算 Softmax 变得非常昂贵。

* 对于大多数语言模型，目标是通过最大化下一个词（或子词）的条件概率来进行训练。具体来说，对于给定的文本序列，大语言模型的目标是最大化下一个词的预测概率。这个过程可以通过 交叉熵损失函数 来表示。

* Autoencoder Models： BERT、T5 是自编码器模型； 模型通过 掩码语言模型（Masked Language Model, MLM）来训练，即随机遮蔽输入的一些词，然后让模型预测这些被遮蔽的词。由于大语言模型的类别数非常庞大（通常是数十万个词汇或子词单元），计算 Softmax 的代价高昂。因此，大语言模型常常使用优化技术（如 采样、层次 Softmax 或 负采样）来加速训练。虽然大语言模型的训练目标本质上是 最大化条件概率，并使用 交叉熵损失，但为了处理庞大的类别数，常常会结合一些优化方法来减少计算复杂度。