# Byte-Pair Encoding
## 是什么
BPE是现代大模型中实现Tokenization（分词）的一种主流方法。

## 为什么需要它
先从问题出发，LLM的目标是输入文本，输出文本，但是模型是通过数学运算运转的，因此需要先将文本转化为数字进行计算。
有几种自然的想法：
1. 一个字分配一个id，缺点是文本序列会变得很长，尤其对于英文文本而言，“hello”会占用5个token。Transformer模型的注意力机制计算复杂度与序列长度的平方成正比（O(N²)），过长的序列会极大地拖慢训练和推理速度。
2. 一个词分配一个id，缺点是词表会变得非常庞大，更致命的是，模型只能理解训练时见过的词。一旦遇到新词、拼写错误、专业术语或网络俚语，模型就束手无策了。

因此需要寻找一种方法，规避上述的缺点。

## 具体是怎么做的
unicode是一个涵盖各语言的庞大字典，定义了UTF-8、UTF-16、UTF-32等编码格式，其中UTF-8因其高效与兼容性强的特点成为互联网的主流。同样的，BPE通常也使用UTF-8进行编码。下面举个例子。
比如要将【hello】进行编码：
1. 先将单词分解成字节流【b'h', b'e', b'l', b'l', b'o'】，对于中文或一些特殊的字符而言，一个字由多个字节编码，一个字符的字节也会被拆分成单独的。
2. 寻找相邻的字节【b'he', b'el', b'll', b'lo'】，并将其中频率最高的提取出来，如果遇到频率最大且相同的，一般会选择第一个遇到的，或者字典序更大的，这里就取b'he'。
3. 更新字节流【b'he', b'l', b'l', b'o'】

重复这样的步骤，维护一个merges表，记录每次合并的规则，比如【b'h', b'e'】，维护一个vocab表，编号0~255对应字节0~255，留下基础字节，后续编号按照合并的字节扩充，比如【256: b'he'】，直到将vocab的数量扩充到目标大小。
通过这样的方式，让序列长度和词表大小都更加可控，并且由于将所有词都先用UTF-8编码，即使遇到没有见过的词，也可以用0~255的基础字节表示。不过从语义上来说，单个token并不一定具有明确含义。

## 实现细节
1. 在处理前要先对文本进行预分词，正确处理标点符号、空格、换行等字符，用一个字典统计`{词: 出现次数}`。
2. 假设目标vocab大小是5000，初始的vocab大小为256个基础字节+特殊token（比如[end_of_sequence]），这里先要统计每个pair的次数，遍历`{词: 出现次数}`，用`{pair: 出现次数}`做统计，找出频率最高的pair。
3. 在merges表中加入频率最高的pair后，需要更新`{pair: 出现次数}`，可以选择再遍历一次`{词: 出现次数}`，并且还要按照merges规则处理词，更好的办法是增量更新，额外维护一个字典，记录`{pair: 词}`，这样我就可以知道每次更新时影响了哪部分词。找到这次更新相关的`{pair: 词}`，遍历受影响的词在`{词: 出现次数}`中的键值对，统计变化的pair和次数，更新在`{pair: 出现次数}`和`{pair: 词}`中，并剔除不再存在的pair，可以高效的完成更新。

# Tokenizer
根据上面的BPE的细节，所谓Tokenizer的作用其实就是
1. 先将文本字节化，按照BPE阶段的merges表合并
2. 按照vocab表找到对应编号（token id），将文本数字化

## 实现细节
1. 同样要先将文本预分词，输入string，输出list
2. 按照merges规则，将每个分词的字节合并，暴力的方法就是遍历merges表，然后遍历整个预分词的list，双重循环处理。更快的方法是维护一个`{pair: list[index]}`的字典，按照merges表获取预分词的索引，按照规则替换预分词列表中的内容，然后统计新的预分词中的pair，更新`{pair: list[index]}`字典，最后在字典中删除这一次merge的pair。
3. 面对大文本要使用流式处理，对于生成的大token序列数据，也要存成一个易于流式处理的格式，便于后续的训练。

# Transformer language model
## pytorch向量与矩阵运算
在线性代数的课本上，向量与矩阵运算的公式一般写为$y = Wx$，其中$x$为列向量，但是在计算机中，$x$一般被创建为行向量。为了与数学上的定义一致，在pytorch中的向量与矩阵相乘被定义为$y = xW^T$。以线性神经网络为例，矩阵实际被存储为out_dim行，in_dim列，只是在计算时会进行转置。

## Relative Positional Embeddings ( RoPE )
RoPE的思路是对每个token在embedding维度上编码，每两个做一次旋转，旋转角度由sequence位置和$d_k$位置决定。像时钟的指针一样去拨动不同位置的隐藏特征，一方面，这可以保证位置编码的不重复，另一方面，不管两个token的绝对位置如何变化，只要相对位置不变，二者相差的角度就不变，增强了稳定性与泛化特性。将绝对位置编码的可并行性、相对位置编码的长度外推能力和零额外参数的高效性结合在了一起。

在实现上，旋转矩阵只与位置有关，因此可以在初始化时预先存好，后续直接索引使用。每对参数的运算公式为$y_out = [x_1*cos - x_2*sin, x_1*sin + x_2*cos]$，因此可以先取出奇数列与偶数列，矩阵运算出结果后拼接返回。

RoPE主要作用于Q和K矩阵上，在点积时参数中会只剩下m-n的相对位置关系，而绝对位置m和n的独立值则在计算中抵消了，让模型天然理解“距离”而非“坐标”。这里的好处在于让位置信息成功嵌入，而不带有绝对位置信息的误导，不至于换个位置就不认识了。同时直接作用于Q和K，保留了embedding空间的原始语义信息，让模型不必学习如何在语义空间中剥离位置信息。

## softmax中的数值稳定性技巧
softmax的公式为
$$softmax(x)_i = \frac{exp(v_i)}{\sum_{j=1}^n exp(v_j)}$$
其中$exp(v_i)$的值域为[0, +inf)，在指数大时容易超出数值上限，为了稳定计算，通常会让$v_i$减去数据中的最大值$v_{max}$，保证指数的范围在(-inf, 0]，从而保证值域不会超限。

## 为什么点积注意力要除以$\sqrt{d_k}$
1. 点积的数值会随维度增大而增长：$QK^T$是在$d_k$维度做内积
2. 过大值使softmax梯度消失：softmax函数对较大的输入值会输出接近0或1的极端概率，此时这些位置的梯度会变得非常小，导致模型难以训练，甚至出现梯度消失。
3. 除以$d_k$起到缩放作用：使点积结果保持在一个合理范围（向0靠近），从而让softmax函数落在一个梯度适中、对输入变化敏感的区域，既稳定了训练，又保留了区分不同注意力权重的敏感性。

## 多头注意力
将$d_k$拆分，以分别学习不同语义特征，拼接后再进入线性层，融合各个特征。
```python
Q = rearrange(Q, "... sq_l (h d) -> ... h sq_l d", h=self.h)
K = rearrange(K, "... sq_l (h d) -> ... h sq_l d", h=self.h)
V = rearrange(V, "... sq_l (h d) -> ... h sq_l d", h=self.h)

Q = self.rope(Q, token_positions=token_positions)
K = self.rope(K, token_positions=token_positions)

attention = scaled_dot_product_attention(Q, K, V, mask=causal_mask)
attention = rearrange(attention, "... h sq_l d -> ... sq_l (h d)", h=self.h)

output = self.O(attention)
```

## transformer层
完整的transformer层 $y = x + MultiHeadSelfAttention(RMSNorm(x))$

## LLM的预训练目标
预训练时，假设训练素材是10个token的句子，那输入就是前9个token，目标就是后9个token。在模型的最后一层，lm_head会将$9*d_k$的矩阵变为$9*vocab_size$，然后计算softmax，将问题变为分类任务，目标是使targets的概率最大。

## cross entroy的数值稳定性技巧
交叉熵作为损失函数，衡量的是两个概率分布之间的差异程度，在这里，将targets的理想概率取1，其他均为0，让模型不断贴近这个分布。

交叉熵要计算-log(softmax(logits))，直接计算会有问题如果x很大，exp(x)会溢出（变成inf），如果x很小且负数，exp(x)会下溢（变成0），然后log(0)得-inf。

在数学上，下式是等价的，所以办法就是转化为右式进行计算，最后取token级的平均值，注重每次预测。
$$log(softmax) = log(\frac{exp(x_i)}{\sum_{j=1}^n exp(x_j)}) = x_i - log(\sum_{j=1}^n exp(x_j))$$

## 优化器内部的结构
|属性名称|数据类型|作用与描述|
|:--|:--|:--|
|`param_groups`|`list[dict]`|参数组管理。这是优化器与模型参数交互的核心结构。列表中的每个字典代表一组参数，包含 `'params'`（具体的参数列表）以及该组专属的超参数（如 `lr`, `weight_decay` 等）。|
|`state`|`dict`|状态管理。用于存储每个参数在更新过程中的状态信息（如动量缓冲区、历史梯度平方和等）。键通常是参数对象本身，值是包含状态张量的字典。|
|`defaults`|`dict`|默认超参数。存储优化器初始化时的默认配置。当某个参数组没有指定特定超参数时，就会从这里获取默认值。|

# 一些LLM行业“模板”
前置归一化；RMSNorm；RoPE；SwiGLU/GeGLU；无Bias
FFN的维度$d_{ff} = 8/3 d_{model}$，注意力头的维度通常为64或128

# 训练结果
在GPU云平台上使用双3090，DDP训练50000步，按照课程建议设置超参数
```python
# hyperparameters
## training loop
'total_steps': 50000,
'batch_size': 32,
'lazy_load': False,

## model
'vocab_size': 10000,
'context_length': 256,
'd_model': 512,
'd_ff': 1344,
'rope_theta': 10000,
'num_layers': 4,
'num_heads': 16,

 ## optimizer
'lr_max': 1e-3,
'lr_min': 1e-6,
'betas': (0.9, 0.999),
'eps': 1e-6,
'weight_decay': 0.01,
'l2_max': 1,
```

总参数量估算：embedding层$10000*512$，每个多头注意力块中有Q、K、V、O四个大小为$512*512$的矩阵，以及SwiGLU中的三个大小为$512*1344$的矩阵，共4层transformer block，再加上输出线性层$10000*512$，总参数量约为22.7M，非embedding参数约17.6M

用wandb监控训练参数，训练过程很稳定，loss下降到2.8左右，perplexity在16.4
![Alt text](https://cdn4.winhlb.com/2026/05/22/6a100196b1444.png)

最后用p=0.1，temperature=1，prompt = "Once upon a time there was a little boy named Ben."看下生成效果：
> 选词平均概率15.50%：Once upon a time there was a little boy named Ben. Every night Amy ran back yard warm blanket oats hurt wing hear sound in puddles everywhere She dreamed of butter begged Greeny want Buddy nosy vine lullaby low price tag with sand sour day Go smile moment old bananas pink slides vest lands cubHoppy firemen chew town time outdoors veterinarian passports away stablehew showing slides hider sunny day moment sweet boy named Max loved him shy pigeon laughing forth sunny day boy named Doggy escape four years old woman named Hoot Hop haircut blue yogurt'd buy silence kiss on sing nicely foods switch much fun playing football fly buzzed forth first child named Toto ordered a mess. She saw many fun learning great festival mother parent low goat named Zip seek safari store clerk barked games every day driving clear yard. She saw many colorful piano shade cage thin slides fast asleep parade tables clouds nod Spot barked happily Ben nod laughing forth low finish line Whiskers met Sue set deeper possibilities hill bigger than seek seek seek safari smile sits forward slides Hoppy lived happily munched stupid stormened anymore. She bought silence price tag, buy night Amy get dark penny splashed everywhere in time ago anymore. She'd buy himself wise old door that night sky bright colours while Amy smile. She saw many wonderful festival low price tag happening to share your imagination.' glad Susie exclaimed Jenny bought
> 
> 选词平均概率13.33%：Once upon a time there was a little boy named Ben. She asks Mr giant muffin soil for hurting much gum caught me untangle shadowsThank you write “Because teddy bear appeared asks Tom smiled lots of tiny helmet?" The wife giggled purr mother rushed boardparcream recorder first I have seen lots of tiny bunny scooped sharing bottle of soft bear appeared says letterbox best friends Charlie bear appeared smiling about how honeycom Woof mine my ladder steady comb anymore. It burned oh no candy wrapper Tim smiled at my violin strip mark idea. The threadBunny met Bunny chases your diary behind a great hunter named Squeaky forgave Max grabbed nothing happens, Sam saw many branches apart, Sue found an old flute bell ring earlier diary mouth watered our favorite mitten my favorite gum bicycletle cap began chopping paintbrush bear appeared there Whenever Speedy raced done the camera oh no cord umbrella bucket OK note umbrella bear away searching for special mineral puddles filthy teddy bear appeared!" This door that no lipstickJohn yelled much fun shapes became fancy veil young hairdresser Have kite ropes there wiped Bobo says yes!" The curtain again she asked Whiskers course “My name coming a seesaw all day long blonde mitten isn't know much fun activity “No mum smiled at the microscope wall perfect lollipop lime there until finally reached the helmet bubbles away all kinds journal bars may drop rabbit hears
> 
> 选词平均概率12.37%：Once upon a time there was a little boy named Ben. Now Let meWhiskers agreed to buy timeBobo want to push Zip seek quietly throw snowballs thinking Brown WilPolly rising seek timeBuddy diveWhiskers especially bandWhiskers every morning springOf course Sally's fix music Whiskers<|endoftext|>

基本上还是狗屁不通，从选词平均概率、loss和perplexity看，模型仍旧是欠拟合的。

# 参考资料
1. [CS336 Spring 2026 Assignment 1: Basics](https://github.com/stanford-cs336/assignment1-basics)
2. [个人实现仓库](https://github.com/MAGMA27/assignment1-basics)