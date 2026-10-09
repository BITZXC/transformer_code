# -*- coding: utf-8 -*-
#/usr/bin/python2
"""
Transformer 模型构建脚本 (基于 kyubyong 的 TensorFlow 实现)
作者: kyubyong park (June 2017)
原项目: https://www.github.com/kyubyong/transformer
本文件: 添加中文注释版本

功能说明:
  构建一个用于德英翻译的 Transformer 模型计算图，
  包括 Encoder、Decoder、损失计算、训练优化器等完整组件。
"""
from __future__ import print_function
import tensorflow as tf

from hyperparams import Hyperparams as hp
from data_load import get_batch_data, load_de_vocab, load_en_vocab
from modules import *
import os, codecs
from tqdm import tqdm


class Graph():
    """
    Graph 类: 封装整个 Transformer 模型的计算图构建逻辑。

    属性:
        graph:            TensorFlow 计算图
        x:                德语输入序列 (batch_size, seq_len)
        y:                英语目标序列 (batch_size, seq_len)
        decoder_inputs:   Decoder 输入 (右移一位 + 起始符)
        enc:              Encoder 输出表示
        dec:              Decoder 输出表示
        logits:           未归一化的 logits (batch_size, seq_len, vocab_size)
        preds:            预测的 token id
        istarget:         有效位置掩码 (非 padding 位置为 1)
        acc:              准确率
        mean_loss:        平均损失
        train_op:         训练操作
        global_step:      全局步数计数器
    """
    def __init__(self, is_training=True):
        """
        初始化 Graph。

        参数:
            is_training: bool, 是否为训练模式。
                         True  时从数据加载器获取 batch 数据;
                         False 时通过 placeholder 接收外部输入(推理阶段)。
        """
        # 创建新的 TensorFlow 计算图
        self.graph = tf.Graph()
        with self.graph.as_default():
            if is_training:
                # 训练模式: 从数据加载器获取一个 batch 的数据
                # self.x: 德语输入, shape=(N, T), N=batch大小, T=序列最大长度
                # self.y: 英语目标输出, shape=(N, T)
                # self.num_batch: 总 batch 数
                self.x, self.y, self.num_batch = get_batch_data()
                # 注意: y 中的值只是词汇表中的索引(整数), 尚未经过 embedding 映射,
                # 因此没有语义信息。例如 cat=5, dog=6, 但 5 和 6 相邻不代表语义相近。

            else:
                # 推理模式: 通过 placeholder 接收外部输入的占位符
                # shape=(None, hp.maxlen) 中 None 表示 batch 大小不固定
                self.x = tf.placeholder(tf.int32, shape=(None, hp.maxlen))
                self.y = tf.placeholder(tf.int32, shape=(None, hp.maxlen))

            # ============================================================
            # 构造 Decoder 输入: 在每句开头插入起始符 <S>(索引=2), 末尾去掉一个 token
            # 实现"右移一位"的效果, 使 Decoder 在每个位置能看到之前的词
            # ============================================================
            # self.y[:, :1]   -> 取第一列(每句第一个token), shape=(N, 1)
            # tf.ones_like    -> 生成全1张量, shape同输入
            # *2              -> 全1乘以2, 得到全2的张量(2是<S>起始符的索引)
            # tf.concat       -> 拼接 [全2的起始符, y去掉最后一列], 沿最后一维
            # 效果: [S, "I", "love", "you", "</s>"] -> [2, S, "I", "love", "you"]
            self.decoder_inputs = tf.concat(
                (tf.ones_like(self.y[:, :1]) * 2, self.y[:, :-1]), -1
            )  # 2 是 <S> 起始符的索引

            # 加载德语和英语的词汇表(索引 <-> 单词 的双向映射)
            de2idx, idx2de = load_de_vocab()    # 德语: 单词->索引, 索引->单词
            en2idx, idx2en = load_en_vocab()    # 英语: 单词->索引, 索引->单词

            # ============================================================
            # Encoder 部分: 将德语输入序列编码为上下文表示
            # ============================================================
            with tf.variable_scope("encoder"):
                # ---------- 1. 词嵌入 (Word Embedding) ----------
                # 将德语 token 的整数索引映射为稠密向量
                # vocab_size=len(de2idx): 词汇表大小
                # num_units=hp.hidden_units: 嵌入向量维度(如512)
                # scale=True: 输出乘以 sqrt(num_units), 与位置编码量级匹配
                # scope="enc_embed": 变量作用域名, 用于隔离变量命名空间
                self.enc = embedding(
                    self.x,
                    vocab_size=len(de2idx),
                    num_units=hp.hidden_units,
                    scale=True,
                    scope="enc_embed"
                )

                # ---------- 2. 构造 padding mask ----------
                # tf.abs(self.enc): 取 embedding 向量的绝对值
                # tf.reduce_sum(..., axis=-1): 沿向量维度求和, padding位置全0, 有效位置>0
                # tf.sign(...): 将非零值变为1, 得到 0/1 mask, shape=(N, T)
                # tf.expand_dims(..., -1): 在最后扩展一维, shape=(N, T, 1), 方便后续广播
                key_masks = tf.expand_dims(
                    tf.sign(tf.reduce_sum(tf.abs(self.enc), axis=-1)), -1
                )

                # ---------- 3. 位置编码 (Positional Encoding) ----------
                # Transformer 的 attention 是排列不变的, 需要注入位置信息
                # 有两种方案可选:
                if hp.sinusoid:
                    # 方案A: 正弦位置编码(论文原版, 固定公式, 不可学习)
                    # 用正弦和余弦函数在不同频率上生成位置向量
                    self.enc += positional_encoding(
                        self.x,
                        num_units=hp.hidden_units,
                        zero_pad=False,
                        scale=False,
                        scope="enc_pe"
                    )
                else:
                    # 方案B: 可学习的位置嵌入(用 embedding 函数实现)
                    # tf.range(tf.shape(self.x)[1]): 生成位置索引 [0, 1, 2, ..., T-1]
                    # tf.expand_dims(..., 0): 变成 (1, T)
                    # tf.tile(..., [N, 1]): 复制 N 份变成 (N, T)
                    # 然后查位置嵌入表, 得到每个位置对应的向量
                    self.enc += embedding(
                        tf.tile(
                            tf.expand_dims(tf.range(tf.shape(self.x)[1]), 0),
                            [tf.shape(self.x)[0], 1]
                        ),
                        vocab_size=hp.maxlen,
                        num_units=hp.hidden_units,
                        zero_pad=False,
                        scale=False,
                        scope="enc_pe"
                    )

                # ---------- 4. 应用 padding mask ----------
                # 将 padding 位置的向量清零, 避免它们参与后续 attention 计算
                self.enc *= key_masks

                # ---------- 5. Dropout ----------
                # 训练时随机将部分神经元输出置0, 防止过拟合
                # rate=hp.dropout_rate: 丢弃比例(如0.1表示丢弃10%)
                # training=tf.convert_to_tensor(is_training): 训练时True(开启), 推理时False(关闭)
                self.enc = tf.layers.dropout(
                    self.enc,
                    rate=hp.dropout_rate,
                    training=tf.convert_to_tensor(is_training)
                )

                # ---------- 6. 堆叠 N 个 Encoder Block ----------
                # 每个 Block 包含: 多头自注意力 + 前馈神经网络
                # hp.num_blocks 通常为6, 即堆叠6层
                for i in range(hp.num_blocks):
                    with tf.variable_scope("num_blocks_{}".format(i)):
                        # --- 6a. Multihead Self-Attention ---
                        # queries=keys=self.enc: 自注意力, 每个词关注序列中所有词
                        # causality=False: 无因果掩码, 可以双向看到全部上下文(与Decoder不同)
                        # num_heads=hp.num_heads: 多头数量(如8头)
                        self.enc = multihead_attention(
                            queries=self.enc,
                            keys=self.enc,
                            num_units=hp.hidden_units,
                            num_heads=hp.num_heads,
                            dropout_rate=hp.dropout_rate,
                            is_training=is_training,
                            causality=False
                        )

                        # --- 6b. Feed Forward Network ---
                        # 两层全连接网络: hidden_units -> 4*hidden_units -> hidden_units
                        # 中间有 ReLU 激活, 增强模型的非线性表达能力
                        self.enc = feedforward(
                            self.enc,
                            num_units=[4 * hp.hidden_units, hp.hidden_units]
                        )

            # ============================================================
            # Decoder 部分: 基于 Encoder 表示和已生成的词, 逐步生成目标序列
            # ============================================================
            with tf.variable_scope("decoder"):
                # ---------- 1. 词嵌入 ----------
                # 对 Decoder 输入(英语token)做词嵌入
                # 注意: 这里用 en2idx 的词汇表(因为是英语输出)
                self.dec = embedding(
                    self.decoder_inputs,
                    vocab_size=len(en2idx),
                    num_units=hp.hidden_units,
                    scale=True,
                    scope="dec_embed"
                )

                # ---------- 2. 构造 padding mask ----------
                key_masks = tf.expand_dims(
                    tf.sign(tf.reduce_sum(tf.abs(self.dec), axis=-1)), -1
                )

                # ---------- 3. 位置编码 ----------
                if hp.sinusoid:
                    # 方案A: 正弦位置编码
                    self.dec += positional_encoding(
                        self.decoder_inputs,
                        vocab_size=hp.maxlen,
                        num_units=hp.hidden_units,
                        zero_pad=False,
                        scale=False,
                        scope="dec_pe"
                    )
                else:
                    # 方案B: 可学习的位置嵌入
                    self.dec += embedding(
                        tf.tile(
                            tf.expand_dims(
                                tf.range(tf.shape(self.decoder_inputs)[1]), 0
                            ),
                            [tf.shape(self.decoder_inputs)[0], 1]
                        ),
                        vocab_size=hp.maxlen,
                        num_units=hp.hidden_units,
                        zero_pad=False,
                        scale=False,
                        scope="dec_pe"
                    )

                # ---------- 4. 应用 padding mask ----------
                self.dec *= key_masks

                # ---------- 5. Dropout ----------
                self.dec = tf.layers.dropout(
                    self.dec,
                    rate=hp.dropout_rate,
                    training=tf.convert_to_tensor(is_training)
                )

                # ---------- 6. 堆叠 N 个 Decoder Block ----------
                for i in range(hp.num_blocks):
                    with tf.variable_scope("num_blocks_{}".format(i)):
                        # --- 6a. Masked Multihead Self-Attention ---
                        # queries=keys=self.dec: 自注意力
                        # causality=True: 有因果掩码(未来信息不可见), 保证自回归生成  
                        # scope="self_attention": 命名作用域
                        self.dec = multihead_attention(
                            queries=self.dec,
                            keys=self.dec,
                            num_units=hp.hidden_units,
                            num_heads=hp.num_heads,
                            dropout_rate=hp.dropout_rate,
                            is_training=is_training,
                            causality=True,#与encoder最大的不同，自注意力加入掩码机制，屏蔽未来信息。
                            scope="self_attention"
                        )

                        # --- 6b. Multihead Attention (Encoder-Decoder Attention) ---
                        # queries=self.dec: Decoder 的当前表示
                        # keys=self.enc: Encoder 的输出表示
                        # causality=False: 可以关注 Encoder 输出的所有位置
                        # 这一步让 Decoder 在生成每个词时"查阅"Encoder 对输入句子的理解
                        self.dec = multihead_attention(
                            queries=self.dec,
                            keys=self.enc,
                            num_units=hp.hidden_units,
                            num_heads=hp.num_heads,
                            dropout_rate=hp.dropout_rate,
                            is_training=is_training,
                            causality=False,
                            scope="vanilla_attention"
                        )

                        # --- 6c. Feed Forward Network ---
                        self.dec = feedforward(
                            self.dec,
                            num_units=[4 * hp.hidden_units, hp.hidden_units]
                        )

            # ============================================================
            # 输出层与损失计算
            # ============================================================

            # ---------- 最终线性投影 ----------
            # 将 Decoder 的输出向量映射到词汇表维度
            # self.dec: (N, T, hidden_units) -> logits: (N, T, vocab_size)，特征向量转换成字符表的打分值
            self.logits = tf.layers.dense(self.dec, len(en2idx))

            # ---------- 预测 ----------
            # tf.argmax(logits, axis=-1): 取概率最大的 token 索引
            # tf.to_int32: 转换为 int32 类型
            self.preds = tf.to_int32(tf.argmax(self.logits, dimension=-1))

            # ---------- 有效位置掩码 ----------
            # tf.not_equal(self.y, 0): 目标序列中不等于0的位置为True(有效位置)
            # padding位置的索引为0, 需要排除在损失计算之外
            # tf.to_float: 转为 float 类型(0.0/1.0)
            self.istarget = tf.to_float(tf.not_equal(self.y, 0))

            # ---------- 准确率计算 ----------#本批次所有
            # tf.equal(self.preds, self.y): 预测正确的位置为True
            # * self.istarget: 只统计有效位置的准确率
            # 最后除以有效位置的总数，acc是标量
            self.acc = tf.reduce_sum(
                tf.to_float(tf.equal(self.preds, self.y)) * self.istarget
            ) / tf.reduce_sum(self.istarget)
            '''
            tf.equal(self.preds, self.y):逐位置比较预测和真实标签。
                 preds: [3, 5, 2, 0, 0]  y:     [3, 4, 2, 0, 0] → [True, False, True, True, True]
            tf.to_float(...)：转成 float [1.0, 0.0, 1.0, 1.0, 1.0]
            *self.istarget 将padding部位也变成0  [1.0, 0.0, 1.0, 0.0, 0.0]
            tf.reduce_sum()/ (tf.reduce_sum(self.istarget)：计算在非 padding 位置上，预测等于真实标签的比例
             '''

            # 将准确率加入 TensorBoard 摘要
            tf.summary.scalar('acc', self.acc)

            # ============================================================
            # 训练相关配置 (仅在训练模式下构建)
            # ============================================================
            if is_training:
                # ---------- 标签平滑 (Label Smoothing) ----------
                # 将硬标签(one-hot)转换为软标签, 防止模型过于自信
                # tf.one_hot(self.y, depth=len(en2idx)): 转 one-hot 编码
                # label_smoothing(...): 对正确类别和错误类别都做轻微扰动
                self.y_smoothed = label_smoothing(
                    tf.one_hot(self.y, depth=len(en2idx))
                )

                # ---------- 损失计算 ----------
                # softmax + 交叉熵损失
                # logits: 模型未归一化的输出（N,T,V）
                # labels: 标签平滑后的软标签(N,T)，
                self.loss = tf.nn.softmax_cross_entropy_with_logits(
                    logits=self.logits, labels=self.y_smoothed
                )

                # 平均损失: 只对有效位置(非padding)求平均，标量  一个batch过完了以后，计算一次loss,
                self.mean_loss = tf.reduce_sum(
                    self.loss * self.istarget
                ) / tf.reduce_sum(self.istarget)  

                # ---------- 优化器配置 ----------
                # 全局步数计数器(不可训练, 仅用于记录训练进度)
                self.global_step = tf.Variable(
                    0, name='global_step', trainable=False
                )


                # Adam 优化器配置
                # learning_rate=hp.lr: 学习率
                # beta1=0.9: 一阶矩估计的指数衰减率
                # beta2=0.98: 二阶矩估计的指数衰减率(Transformers论文推荐值)
                # epsilon=1e-8: 数值稳定性参数
                self.optimizer = tf.train.AdamOptimizer(
                    learning_rate=hp.lr, beta1=0.9, beta2=0.98, epsilon=1e-8
                )

                # 训练操作: 最小化平均损失, 同时更新全局步数
                self.train_op = self.optimizer.minimize(
                    self.mean_loss, global_step=self.global_step
                )

                # ---------- TensorBoard 摘要 ----------
                tf.summary.scalar('mean_loss', self.mean_loss)
                # 合并所有摘要操作(方便一次性写入 TensorBoard)
                self.merged = tf.summary.merge_all()


# ============================================================
# 主程序: 训练流程
# 每执行一次 sess.run(train_op)，就对应处理一个 Batch 的数据，并完成一次完整的“前向传播 + 反向传播 + 参数更新”
# ============================================================
if __name__ == '__main__':
    # 加载词汇表
    de2idx, idx2de = load_de_vocab()    # 德语词汇表
    en2idx, idx2en = load_en_vocab()    # 英语词汇表

    # 构建计算图 (训练模式)
    g = Graph("train")
    print("Graph loaded")

    # 创建 TensorFlow 会话管理器
    # sv = Supervisor 会自动处理:
    #   - 创建/恢复会话
    #   - 初始化变量
    #   - 保存检查点
    #   - 管理 TensorBoard 日志
    # logdir=hp.logdir: 日志和检查点保存目录
    # save_model_secs=0: 不自动定时保存(由手动控制)
    sv = tf.train.Supervisor(
        graph=g.graph,
        logdir=hp.logdir,
        save_model_secs=0
    )

    # 在托管会话中运行训练循环
    with sv.managed_session() as sess:
        # 遍历所有 epoch
        for epoch in range(1, hp.num_epochs + 1):
            if sv.should_stop():
                break

            # 遍历每个 batch
            # tqdm: 进度条显示
            # g.num_batch: 总 batch 数
            for step in tqdm(
                range(g.num_batch),
                total=g.num_batch,
                ncols=70,
                leave=False,
                unit='b'
            ):
                # 执行一步训练: 前向传播 + 反向传播 + 参数更新
                sess.run(g.train_op)

            # 获取当前全局步数
            gs = sess.run(g.global_step)

            # 每个 epoch 结束后保存模型检查点
            # 文件名包含 epoch 号和全局步数
            sv.saver.save(
                sess,
                hp.logdir + '/model_epoch_%02d_gs_%d' % (epoch, gs)
            )

    print("Done")
