# Introduction
The CPU usage (also include RAM and network) of a server in a large-scale cluster are virtually fixed at a fixed time. 
Aberrant CPU usage often indicates an abnormal machine
state, which must be detected and fixed.
As we are monitoring large number of machines, it's
preferable to use automatic anomaly detection tools.
This simple snippet demonstrates the use of Naive Bayes
Model for anomaly machine state detection.
Compared with discriminative models, generative models
is more suitable for this task because the distribution of
signals can be best modeled by a Gaussian distribution -- no
training is needed.
Naive Bayes Model provides additional flexibility for this task
as the user can dictate the number of features (time interval)
for monitoring.



# Implementation Details

服务器状态异常监测的方法是每隔一定的时间, 提取这一段时间的CPU使用数据,然后根据朴素贝叶斯模型判断是否有异常发生.

假定每隔10秒钟取一次CPU的使用数据，那么现在就有 t0-t9 的CPU使用率的数据 c0-c9.

已知CPU的状态有两种s = {s0, s1}, 其中s0表示无异常, s1表示有异常.根据贝叶斯理论, 知道了机器发生异常的先验概率(Prior)(比如100台机器通常有1台坏掉的, 那么可以认为 P(s = s1) = 1 / 100 = 0. 1), 现在我们有了 t0-t9 的CPU使用率的数据 c0-c9, 称之为证据(Evidence), 这时候需要根据证据来更新先验概率,更新后的概率称之为后验概率(Posterior). 更新方法是使用贝叶斯公式 posterior = prior * likelihood / evidence,把这个公式放在本场景内是这样的P(s | c0-c9) = P(s) * P(c0-c9 | s) / P(c0-c9),其中P(s | c0-c9)是后验概率, P(s)是先验概率,   P(c0-c9|s)是似然性(Likelihood),P(c0-c9)是证据发生的概率.现在P(s)已知,关键是计算P(c0-c9 | s)和P(c0-c9).首先根据概率公式, P(c0-c9) = P(c0-c9, s0) + P(c0-c9, s1) = P(c0-c9 | s0) * P(s0) + P(c0-c9 | s1) * P(s1),所以有了P(c0-c9 | s), P(c0-c9)就能根据上述公式计算出来.所以问题的关键是计算P(c0-c9 | s).

考虑机器正常情况下的 P(c0-c9 | s = s0). 这时候P(c0-c9 | s =s0),表示的是机器在正常s = s0的条件下,CPU的使用率是c0-c9的概率. 根据Naive Bayes的假设: 给定机器的状态, CPU在不同时间点的使用率是相互独立的. 在这样的假设下P(c0-c9 | s = s0) = P(c0 | s = s0) * p(c1| s = s0) * . . . * p(c9 | s = s0). 所以关键是计算P(c | s). 以P(c0 | s = s0)为例, P(c0 | s = s0)表示的是机器在正常的状态下在t0时刻CPU的使用率是c0的概率. 机器正常时在某一时刻的CPU的使用率都是相对固定的, 因此CPU的使用率服从高斯分布的. P(c0 | s = s0)就可以利用高斯分布的概率密度函数计算出来. 高斯分布的参数(均值和标准差)是通过对以往CPU使用率的极大似然估计来得出的.

考虑机器异常情况下的P(c0-c9 | s = s1). 这时候P(c0-c9 | s =s0)表示的是机器在异常s = s1的条件下CPU的使用率是c0-c9的概率.在异常的情况下采用均匀分布, 表示机器在异常情况下, CPU的使用率在0到1的范围内是等可能的.不过考虑到机器异常时CPU的使用率通常较高, 可以考虑更合理的模型.

当P(s = s1 | c0-c9)大于0.5时，就可以判断机器在时间段t0-t9出现了异常.


