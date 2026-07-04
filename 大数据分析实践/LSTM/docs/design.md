# 基于LSTM-Attention的NBA球员能力等级预测系统

副标题：

**利用Per36标准化统计指标预测球员下一赛季能力等级**

---

# 一、实验目标

构建一个面向球队管理层的球员能力评估系统。

输入：

```text
过去3个赛季表现
```

输出：

```text
下一赛季能力等级
```

应用场景：

* 球员培养
* 引援评估
* 续约决策
* 替补球员挖掘

---

# 二、研究思路

传统统计：

```text
直接比较场均得分
```

缺点：

```text
受出场时间影响
```

---

本实验：

采用：

```text
Per36标准化指标
```

即：

```python
PTS36 = PTS / MP * 36
AST36 = AST / MP * 36
TRB36 = TRB / MP * 36
STL36 = STL / MP * 36
BLK36 = BLK / MP * 36
TOV36 = TOV / MP * 36
```

使模型学习：

```text
球员能力变化
```

而非：

```text
球队给予的出场机会变化
```

下面这版是我认为比较合理、并且符合篮球数据分析逻辑的数据预处理方案，可以直接放进你的课程设计报告。

---

# 4 数据预处理

## 4.1 数据清洗

### （1）删除重复赛季记录

NBA数据集中存在如下情况：

| Year | Player       | Tm  |
| ---- | ------------ | --- |
| 2010 | LeBron James | MIA |
| 2010 | LeBron James | CLE |
| 2010 | LeBron James | TOT |

其中：

```text
TOT = 球员当赛季所有球队数据汇总
```

为了保证：

```text
一个球员
一个赛季
对应唯一记录
```

保留：

```python
Tm == "TOT"
```

删除同赛季分球队记录。

对于只效力一支球队的球员则直接保留原记录。

---

### （2）缺失值处理

数据集中缺失值主要出现在：

```python
FG%
3P%
FT%
```

进一步检查发现：

```python
3PA == 0
```

对应：

```python
3P% == NaN
```

或者：

```python
3P% == 0
```

两种情况同时存在。

这说明：

```text
缺失值本质表示未进行投篮尝试
```

而非数据丢失。

因此采用统一处理：

```python
FG% = FG%.fillna(0)
3P% = 3P%.fillna(0)
FT% = FT%.fillna(0)
```

但后续特征工程阶段不直接使用这些命中率指标。

---

## 4.2 特征工程

### （1）构造场均出场时间

原始数据包含：

```python
G
MP
```

其中：

```python
MP = 赛季总出场时间
```

为了消除赛季长度影响，构造：

```python
MPG = MP / G
```

表示：

```text
场均出场时间
```

反映：

```text
教练信任度
球队角色定位
```

---

### （2）构造 Per36 指标

由于：

```python
PTS
AST
TRB
```

等累计统计量与出场时间高度相关。

例如：

```python
PTS ≈ 能力 × MP
```

因此采用 NBA 数据分析领域广泛使用的：

```text
Per36 Statistics
```

进行标准化。

计算公式：

```python
Per36 = Stat / MP * 36
```

构造：

```python
PTS36
AST36
TRB36
STL36
BLK36
TOV36
```

以及：

```python
FG36
FGA36

3P36
3PA36

FT36
FTA36
```

其含义为：

```text
如果球员打满36分钟
预计能够获得的统计数据
```

这样能够最大程度消除出场时间差异。

---

### （3）构造高级篮球指标

#### True Shooting Percentage（TS%）

用于综合评价球员得分效率。

计算公式：

```python
TS%

=
PTS

/

(
2 × (FGA + 0.44 × FTA)
)
```

反映：

```text
真实得分效率
```

---

#### Assist-Turnover Ratio

计算公式：

```python
AST_TOV

=
AST36 / TOV36
```

反映：

```text
组织能力
控球稳定性
```

---

#### Three Point Rate

计算公式：

```python
3P_Rate

=
3PA36 / FGA36
```

反映：

```text
外线投射倾向
```

用于区分：

```text
传统内线
现代空间型球员
```

---

## 4.3 特征筛选

### 删除原始累计数据

删除：

```python
PTS
AST
TRB
STL
BLK
TOV

FG
FGA

3P
3PA

FT
FTA

MP
G
```

原因：

```text
这些特征已经被Per36指标替代
```

保留会造成严重信息冗余。

---

### 删除命中率指标

删除：

```python
FG%
3P%
FT%
```

原因：

命中率本质属于：

```python
FG / FGA

3P / 3PA

FT / FTA
```

已经包含在：

```python
FG36
FGA36

3P36
3PA36

FT36
FTA36
```

之中。

保留会增加冗余特征。

---

## 4.4 最终输入特征

最终输入特征如下：

### 基础特征

```python
Age
MPG
```

---

### 投篮能力

```python
FG36
FGA36

3P36
3PA36

FT36
FTA36
```

---

### 比赛贡献

```python
AST36
TRB36

STL36
BLK36

TOV36
```

---

### 高级指标

```python
TS%

AST_TOV

3P_Rate
```

---

最终输入维度：

```python
16维
```

即：

```python
[
Age,
MPG,

FG36,
FGA36,

3P36,
3PA36,

FT36,
FTA36,

AST36,
TRB36,

STL36,
BLK36,

TOV36,

TS%,
AST_TOV,
3P_Rate
]
```

---

## 4.5 标准化处理

采用：

```python
StandardScaler
```

标准化所有连续特征。

处理流程：

```text
训练集
    ↓
fit scaler
    ↓
训练集 transform
验证集 transform
测试集 transform
```

避免测试集信息泄露。

---

## 4.6 时间序列样本构建

按照：

```python
Player
Year
```

排序。

每名球员形成职业生涯序列：

```text
Season1
Season2
Season3
...
SeasonN
```

设置：

```python
SEQ_LEN = 3
```

采用滑动窗口生成样本。

例如：

```text
2009
2010
2011
    ↓
2012
```

构造：

```python
X = [
    season_2009,
    season_2010,
    season_2011
]

y = label_2012
```

最终形成：

```python
X.shape = (N, 3, 16)

y.shape = (N,)
```

作为 LSTM-Attention 模型输入。

---




# 四、序列样本构建

## Step1

按：

```python
Player
Year
```

排序。

---

例如：

```text
Curry

2009
2010
2011
2012
2013
2014
```

---

## Step2

滑动窗口

设置：

```python
SEQ_LEN = 3
```

---

样本1

输入：

```text
2009
2010
2011
```

输出：

```text
2012
```

---

样本2

输入：

```text
2010
2011
2012
```

输出：

```text
2013
```

---

最终：

```python
X.shape

(N,3,12)
```

---

# 五、标签设计

采用：

```text
下一赛季PTS36等级
```

---

划分标准：

| 等级 | PTS36 |
| -- | ----- |
| 0  | <12   |
| 1  | 12~18 |
| 2  | 18~25 |
| 3  | >25   |

---

对应：

| 等级名称            |
| --------------- |
| Role Player     |
| Rotation Player |
| Star            |
| Superstar       |

---

预测目标：

```python
y ∈ {0,1,2,3}
```

---

# 六、数据集划分

采用时间划分。

训练集：

```text
1980-2012
```

验证集：

```text
2013-2015
```

测试集：

```text
2016-2017
```

---

原因：

```text
符合真实未来预测场景
```

避免：

```text
未来信息泄露
```

---

# 七、模型设计

## Baseline

MLP

```text
Input

↓

FC(64)

↓

FC(32)

↓

Output
```

---

## Model1

LSTM

```text
Input

↓

LSTM(64)

↓

LSTM(32)

↓

FC

↓

Output
```

---

## Model2

LSTM + Attention

```text
Input

↓

LSTM

↓

Attention

↓

FC

↓

Softmax
```

---

Attention作用：

自动学习：

```text
过去哪个赛季最重要
```

例如：

```text
Season1 0.15

Season2 0.30

Season3 0.55
```

---

# 八、损失函数

分类任务：

```python
CrossEntropyLoss
```

---

优化器：

```python
Adam
```

---

学习率：

```python
1e-3
```

---

Batch Size：

```python
64
```

---

Epoch：

```python
100
```

---

EarlyStopping：

```python
patience=10
```

---

# 九、评价指标

## 分类指标

* Accuracy
* Precision
* Recall
* F1-Score

---

## 可视化

### Loss Curve

训练损失曲线

---

### Accuracy Curve

准确率曲线

---

### Confusion Matrix

混淆矩阵

---

### Attention Heatmap

注意力热力图

展示：

```text
模型最关注哪个历史赛季
```

---

# 十、扩展实验（加分项）

利用训练好的 LSTM Hidden State 作为球员向量：

```python
Player Embedding
```

然后计算：

```python
Cosine Similarity
```

实现：

## 球员替代者发现系统

输入：

```text
Stephen Curry
```

输出：

```text
Damian Lillard
Trae Young
Kyrie Irving
...
```

---

# 十一、预期结论

验证以下假设：

### H1

LSTM优于MLP。

### H2

Attention优于普通LSTM。

### H3

Per36指标比场均指标更能反映球员真实能力。

### H4

球员过去3个赛季数据能够有效预测下一赛季能力等级。

---

这一版比最初的“预测PPG”方案更符合篮球分析逻辑，也更容易在答辩时解释为什么这样设计特征和标签。
