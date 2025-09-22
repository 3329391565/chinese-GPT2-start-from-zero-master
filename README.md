Training GPT2 Chinese from zero to hero
==

1.Description:
---
从头训练一个约82M参数的中文GPT2模型，使用BERT的Tokenizer。中文语料采用《凡人修仙传》小说部分章节，大小约16M。训练15个周期，batchsize=2，梯度累计步数为4（等效batchsize=8），使用stride=512切分数据，启用fp16加速。最终模型能够续写《凡人修仙传》小说，生成10句以上的连续文本。

2.Start:
----
(1)***environment***

首先，我们下载依赖。
```bash
pip install -r requirements.txt
```

(2)***dataset***

准备中文语料，放置在./data/文件夹下，将语料由.txt文件更改为input.json文件
《凡人修仙传》小说语料来源于https://qweree.cn/index.php/282/

按照参考样例./train.json更改input.json文件格式,由于数据集内容为原始的小说内容，包含着大量的非法字符和json读取不支持的控制字符，因此我们对原始数据集文件进行处理，去除其中非法字符，生成预处理好的数据集文件train.json。
```bash
python data\clr_ctrl.py
```

(3)***Model***

在model_config 定义初始GPT-2模型的超参数配置，
- "initializer_range": 0.02 ： 定义了模型参数（如权重矩阵）在初始化时的标准差，权重会在均值为0，标准差为0.02的正态分布中进行随机初始化。
- "layer_norm_epsilon": 1e-05 ： 用于层归一化的常数，用于避免在归一化过程中出现除以零的情况。设置值为1e-05，用于稳定训练。
- "n_ctx": 1024 ： 表示模型上下文窗口的大小，GPT-2 在生成文本时会考虑的最大序列长度。最大长度设为1024，即模型一次最多能处理1024个 token。
- "n_embd": 768 ： 表示每个token的嵌入维度大小，即模型中词向量的维度。设置为768，即每个词汇的表示向量是768维的。
- "n_head": 12 ： 表示自注意力机制中的注意力头的数量。设置为12，即模型的多头注意力机制中有12个独立的头。
- "n_layer": 10 ： 表示 Transformer 编码器中的层数。在这里，设置为 12，即模型有 12 层堆叠的 Transformer 块。
- "n_positions": 1024 ： 表示模型可以处理的最大位置索引，即序列中的最大位置数。最大位置数为 1024，和 n_ctx一致，表示模型最多能处理1024个位置的token。
- "vocab_size": 13317 ： 表示词汇表的大小，即模型可以识别和生成的词汇数量。在这里，词汇表大小为 21128，表示该模型可以处理的词汇量为21128个不同的 token。


(4)***Training***

现在，我们可以使用我们处理好的数据集来训练我们的初始gpt2模型，使用如下命令：
```bash
python train.py --model_config config/model_config_small.json --tokenized_data_path data/tokenized/ --tokenizer_path cache/vocab_small.txt --raw_data_path data/train.json --epochs 15 --log_step 200 --stride 512 --output_dir model/ --device 0 --num_pieces 100 --raw --batch_size 2 --gradient_accumulation 4 --fp16
```

在这个过程中，我们可以看到命令窗口打印出模型的config文件，定义了模型的结构；同时也打印出了模型的参数量，为81894144，约82M

Print Model config
config:
{      
  "attn_pdrop": 0.1,
  "embd_pdrop": 0.1,
  "finetuning_task": null,
  "initializer_range": 0.02,
  "layer_norm_epsilon": 1e-05,
  "n_ctx": 1024,
  "n_embd": 768,
  "n_head": 12,
  "n_layer": 10,
  "n_positions": 1024,
  "num_labels": 1,
  "output_attentions": false,
  "output_hidden_states": false,
  "output_past": true,
  "pruned_heads": {},
  "resid_pdrop": 0.1,
  "summary_activation": null,
  "summary_first_dropout": 0.1,
  "summary_proj_to_labels": true,
  "summary_type": "cls_index",
  "summary_use_proj": true,
  "torchscript": false,
  "use_bfloat16": false,
  "vocab_size": 13317
}
number of parameters: 81894144

训练过程中，每个epoch对应的模型都将存储在./model/目录下，最终训练好的模型将存储在./model/final_model/路径中。

(5)***Generate***

现在，我们可以使用我们用目标语料训练生成的模型来进行文字生成，使用如下命令：
```bash
python generate.py --device 0 --length 1000 --tokenizer_path cache/vocab_small.txt --model_path model/final_model --prefix '[CLS]“砰”的一声' --topp 1 --temperature 1.0 --save_samples --save_samples_path "./mnt/samples.txt"
```


3.Result
--
最终会生成10个文字样本，存储在./mnt/目录下，其中之一如下：

======================================== SAMPLE 1 ========================================

“砰”的一声，一阵热风从远处滚滚而来，随之黑雾一阵翻滚，竟从里面立刻幻化成了一头巨大的双头黑蛟。四只八翅膀只是略一盘旋，就化为一道绿虹的激射而走，只是一个闪动，就消失在了天边尽头处。在原来拦阻的韩立，刚才一击虽然全力催使黑冥甲大汉多半威能，已经将黑冥雾再次召唤而出，但是身为合体后期的魔族尊者，自然丝毫畏惧没有的肆意催动手中巨型狼牙棒的狂攻上去。只见这一次，那些巨型狼牙棒和巨型乌蛟的巨型金锤就硬生生冲出，化为一团乌云的向逃走的黑冥卫更是一阵狂风呼啸而去。以黑冥卫的胆量，这一次韩立和蟹道人联手也不过只有暂时一二了。青龙上人见此情形，脸色自然越发的阴沉了。倒是韩立虽然并未真正坏掉他的好事，但刚才一击，他自然也曾经察觉到二者的可怕。只是这名魔族尊者身上不知何时多出了一名黄袍，并且身上气息恐怖到极点，一身法力凝聚一体后，就再无先前那十余只黄色巨龟。这些巨龟每一只都有金灿灿的异常简单，身上丝毫生命波动没有，但甲上那些符文在这一击下，竟化为丝丝未动，反而一个个闪动下，纷纷放出一道道道黑色剑光出来。而这些巨兽则身上气息纵然强大的不凡，但在那金色剑光强横身躯和巨锤一接触下，却不由自主的纷纷的溃散而灭。下一刻，这些巨鹰上空波动一起，并无一丝满意的从中激射而出。第两千一百九十一章力压黑甲大汉一声冷哼，一根手指粗长剑光一闪而逝，刹那间变得刺目刺目耀眼，同时一片黑色丝丝从眼前一卷而出。心处很明白的同一时间，倚天城中的众多高阶存在均为之色变。刹那间，一层无形光幕丝无声无息的浮现而出，并往四面八方狂散而去，所过之处虚空，不过是一片片平淡无边的白玉般殿堂，一栋栋高约十丈的高台，耸立在那里。而在这片建筑中，隐约有一座三角形的雕像，雕像阁楼，密密麻麻之下，几乎遍布整座大殿的各处地方。“戈同！”“雕像阁”三十六岁的黑甲大汉，失声叫道。浑身气息强大，似乎还在那青冥在上面也算对手的样子。“哼，区区一名合体修士，竟然敢在本座面前张声势，还真是做不到的事情。”青色麒麟低首看了看天空，目中闪过一丝怒色，但口中却冷笑了一声。“算了，我也承认是阁下当年修为大进，本座当年在其他界面前根本不屑一顾的。”巨大锤影响起的越发威胁，青色巨锤在下方一出现的瞬间，就发出阵阵低沉尖鸣声的奔青龙一砸而去。青龙上人听到这里，目中闪过一丝讥讽，但当看到青龙上人的一丝笑意后，心中的忌惮当时，却丝毫不觉到了。只见他身上气息一变，大半身躯赫然已经缩小成了巴

==========================================================================================
