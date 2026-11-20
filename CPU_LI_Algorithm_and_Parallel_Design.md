# CPU 侧 Lightning Indexer：算法、精度与并行实现详解

日期：2026-11-09  
适用项目：LI Offload Evaluation Project 2.1，原生 CPU ABI 3。  
依据：项目中的 native/cpu_li.cpp、native/histogram.hpp、cpu_api.py、caseio.py、scan_cpu.py、topology.py、validation.py。

本文描述实际已实现的 CPU 路径，不把建议中的优化当作现有功能。目标机主路径使用 BF16 输入、`svebf16` Score（SVE2 `svbfdot_f32`）或编译期选择的 `sme` Score（SME `svmopa_za32_f32_m`）、SVE2 HISTSEG TopK，并显式选择 unordered；`simd` 仍是代码里的默认后端，`blas` 需编译时开启，`sme` 需用 `scripts/build_cpu.sh --sme` 开启。`cpu_bench.py`/`scan_cpu.py` 默认 `order=unordered`；直接调用 `CPUOp` 时默认是 `sorted`，应显式指定。

## 1. LI 的任务与计算边界

Lightning Indexer（LI，闪电索引器）根据当前 Query 和历史 Indexer Key 的相关性，选择最相关的 K 个 Key token 位置，供后续稀疏 Attention 使用。

本实现接收已经生成的 Q、Key、Weights，不包含产生它们的线性投影、RoPE、量化模型部署流程，也不执行后续 Attention。这里的 Indexer Key 是索引器使用的特征，不能直接当作完整 Attention KV Cache 的全部内容。

对于请求 b、Query token i 和可见 Key token j：

$$
u_{b,i,h,j}=\sum_{d=0}^{D-1}Q_{b,i,h,d}\,Key_{b,j,0,d}
$$

$$
score_{b,i,j}=\sum_{h=0}^{H-1}W_{b,i,h}\,\max(u_{b,i,h,j},0)
$$

最后沿 Key token 维度 j 选择最大的 K 个 score 对应的索引。

- H 是 Query 的 Indexer head 数；项目中 H=N1。
- N2=1：所有 Query head 共享一个 Key head。
- W 是每个 Query、每个 head 的动态加权系数，shape 为 [B,S1,H]；它不是生成这些系数的线性层参数矩阵。
- ReLU 作用于每个 head 的 QK 点积，先 ReLU，再乘 W，最后跨 head 求和。
- W 可以为负，因此最终 score 可以为负；不能假定所有得分非负。
- K 在本文表示输出 TopK 数量；Key 表示输入张量，避免二者混淆。

## 2. 输入、输出和数据精度

### 2.1 主要维度

| 符号 | 含义 |
|---|---|
| B | 请求数，即 batch size |
| S1 | 每个请求的 Query 存储长度 |
| T | 项目运行时 Score 的 Key 长度，取 max(1,max(ak)) |
| H / N1 | Query Indexer head 数 |
| N2 | Key head 数，当前固定为 1 |
| D | head dimension，当前主路径固定为 128 |
| K | 每行输出的索引槽位数 |
| bs | PA 存储中每个 block 的 token 数 |
| P | 本次调用请求的 OpenMP 工作线程数 |
| R | 有效 Query 行数，即 sum(aq)，不是 B×S1×H |
| G | TopK 使用的线程组数 |

### 2.2 输入张量

| 输入 | 形状 | 存储格式 | 作用 |
|---|---|---|---|
| Q | [B,S1,H,D] | BF16，每元素 2 字节 | Query 特征，BSND 布局 |
| Key | [block_num,bs,1,D] | BF16，每元素 2 字节 | 按 PA 物理块存储的 Key |
| W | [B,S1,H] | BF16，每元素 2 字节 | head 加权系数 |
| aq | [B] | INT32 | 每个请求的有效 Query 数 |
| ak | [B] | INT32 | 每个请求的有效 Key 数 |
| block_table | [B,max_blocks] | INT32 | 从逻辑 Key block 到物理 block 的映射 |

当前 Python Case 主路径校验 Q=BSND、Key=PA_BSND、Q/Key/W 均为 BF16，D=128，H 属于 8、16、24、32、64。PA block_size 为 16 的倍数，范围为 16～1024。

TopK 数量支持 1～2048，以及 3072、4096、5120、6144、7168、8192。这是本项目与目标原始 LI 接口对齐的限制，不是 radix 算法本身的数学限制。

原生 C++ 另外留有连续 Key 布局和 FP16 解码分支，但当前 Case/CPUOp 正式入口固定走上述 BF16 PA 路径；本文不把低层分支当作已经开放的上层功能。

### 2.3 输出和内部结果

| 数据 | 形状 | 类型 | 说明 |
|---|---|---|---|
| indices | [B,S1,1,K] | INT32 | 逻辑 Key token 索引；不是物理 PA 地址 |
| scores | [B,S1,T] | FP32 | 完整 Score 缓冲区；用于 TopK 和诊断 |
| Q 工作副本 | [B,S1,H,D] | FP32 | 每次调用从 BF16 转换 |
| W 工作副本 | [B,S1,H] | FP32 | 每次调用从 BF16 转换 |
| 排序键 | 候选数长度 | UINT32 | 由 FP32 位模式生成的保序编码 |
| 字节数据 | 候选数长度 | UINT8 | 当前 radix 轮次提取的一个字节 |
| 直方图 | [256]，每工作线程私有 | UINT32 | 每个桶的计数 |
| timing | [4] | C++ double / Python float64 | 原生返回毫秒，benchmark 转换成微秒 |

主输出是 indices。CPUOp.run() 同时返回复用的 scores 和 timing 数组；这些对象会被下次调用改写，若要长期保留某次结果，应复制。

无效或被 mask 的 Score 槽位初始化为 -∞；无效索引槽位初始化为 -1。有效计算产生 NaN 或无穷大时，内核返回错误，不把这类值当作普通可排序得分。

### 2.4 BF16→FP32 的含义

CPU 文件中的 BF16 以 uint16 位模式保存。转换时：

~~~cpp
uint32_t bits32 = uint32_t(bits16) << 16;
float value;
std::memcpy(&value, &bits32, 4);
~~~

对有限 BF16 数值，这个提升是精确的，不会重新获得生成输入时舍弃的精度。生成器先将 FP32 随机数按最近偶数舍入到 BF16；随后 CPU 使用这些 BF16 数值的精确 FP32 表示执行运算。

`simd` 与 `blas` 路径把 BF16 输入提升为 FP32 后计算（FP32 点积 / FP32 SGEMM）；`svebf16` 路径直接对 BF16 位模式执行 `svbfdot_f32`（bf16 乘、fp32 累加）；`sme` 路径把 BF16 提升为 FP32 后用 `svmopa_za32_f32_m` 外积累加 QKᵀ（fp32 乘加）。SVE 与 SME 通过构建宏 `LI_SCORE_USE_SME` 二选一。各类后端与 NPU Cube 的计算顺序与舍入行为可能不同，即使输入相同、结果缓冲区均为 FP32，也不保证逐位一致。

## 3. decode_64k 的具体例子

| 参数 | 值 |
|---|---:|
| B / S1 | 1 / 1 |
| T | 65536 |
| H / N2 / D | 64 / 1 / 128 |
| K | 2048 |
| PA block_size / block_num | 128 / 512 |
| Q shape | [1,1,64,128] |
| Key shape | [512,128,1,128] |
| W shape | [1,1,64] |
| 输出 shape | [1,1,1,2048] |

该用例得到 65536 个最终得分，再执行一次 TopK。64 个 head 合并成一行得分，不是 64 行 TopK。

主要数据量：

- BF16 Key：65536×128×2 = 16 MiB。
- BF16 Q：64×128×2 = 16 KiB。
- BF16 W：64×2 = 128 B。
- FP32 Score：65536×4 = 256 KiB。
- INT32 输出：2048×4 = 8 KiB。
- QK 点积约包含 65536×64×128 = 536870912 个乘加项；按一个乘法和一个加法各计一次操作，约 1.074 GFLOP，不含其他处理。

它采用 DeepSeek-V3.2 的代表形状，但输入由随机数生成，精度为本项目的 BF16 路径；不是完整模型执行，也不是官方 FP8/scale 路径复现。

## 4. 总体执行流程

~~~mermaid
flowchart TD
    A["BF16输入与PA块表"] --> B["可见长度、任务构造与初始化"]
    B --> C["并行FP32 Score计算"]
    C --> D["UINT32保序编码与四轮radix阈值查找"]
    D --> E["GT/EQ筛选与并行输出"]
    E --> F{"输出顺序"}
    F --> G["unordered：直接返回"]
    F --> H["sorted：对选中项追加排序"]
~~~

Score 与 TopK 使用同一个 OpenMP 并行区，二者之间有同步；当前没有跨阶段的流式重叠。完整 Score 先计算完成，再开始 TopK。

## 5. 步骤一：检查输入并建立逻辑 Key 地址

### 输入

Q/Key/W 的 BF16 指针、aq/ak、block_table、维度和执行参数。

### 处理

Python Case 检查元数据、文件字节数、哈希、shape、有效长度和 PA 块编号。原生函数进入后还会检查参数、有效长度、PA 表容量及活动物理块是否越界。

对于逻辑 Key token j：

~~~text
logical_block = j // bs
offset_in_block = j % bs
physical_block = block_table[b, logical_block]
physical_token = physical_block * bs + offset_in_block
Key 向量起始元素 = physical_token * D
~~~

### 输出

每次访问 Key 时可据此定位物理 BF16 向量。正式计算不先把完整 PA Key 重新 gather 成连续大张量，也不为每个 head 复制完整 Key。

### 示例

若 bs=128，block_table[b,1]=7，逻辑 j=130：

~~~text
逻辑块 = 1，块内偏移 = 2
物理 token = 7×128+2 = 898
读取 Key[7,2,0,:]
~~~

最终输出仍是逻辑索引 130，而不是 898。

## 6. 步骤二：计算可见长度并构造任务

### 6.1 Mask 和有效 Query

请求 b 只有 i<aq[b] 的 Query 参与计算。

mode=0：

$$
v(b,i)=ak[b]
$$

mode=3，right-down causal：

$$
v(b,i)=\max(0,ak[b]-aq[b]+i+1)
$$

有效 Key 索引为 0≤j<v(b,i)。这里 ak 表示当前可用 Key 的总长度，decode 用例中包含已经写入 cache 的当前位置。

例如 aq=3、ak=10，三个 Query 的可见 Key 数依次为 8、9、10。

### 6.2 Row：TopK 工作单位

每个有效 Query 对应一条 Row，保存 b、i、可见长度 v，以及展平行号 b×S1+i。有效行数 R=sum(aq)。

### 6.3 Task：Score 工作单位

一个 Task 保存：

~~~text
请求 b
Query 区间 [q0,q1)，最多 query_tile 个
逻辑 Key 区间 [start,end)，最多 tile 个
~~~

同一 Query 分块按最后一条 Query 的最大可见长度生成 Key 任务，具体计算时再跳过各 Query 不可见的位置。

decode_64k 在 tile=128、query_tile=1 时，产生 65536/128=512 个 Score 任务。即使 B=1、S1=1，也有足够多的 Key 分块可供多线程分工。

这与仅对 Batch 循环并行不同：只按 Batch 分工时 B=1 只能有效使用一个工作线程。

## 7. 步骤三：准备 FP32 数据和输出缓冲区

### 输入

BF16 Q/W、已经分配的 FP32 Score 和 INT32 indices。

### 处理

一个 OpenMP 并行区内通过多个静态分工循环完成：

1. Q 全部提升到共享 FP32 工作副本。
2. W 全部提升到共享 FP32 工作副本。
3. Score 全部初始化为 -∞。
4. indices 全部初始化为 -1。

前几个循环使用 nowait 减少不必要的单独等待，最后的同步保证所有初始化结束后再开始 Score 计算。

### 输出

可供所有工作线程只读的 FP32 Q/W，以及初始化完成的 Score、indices。

Key 不在这一步全量转 FP32；它在各 Score 任务内部按需转换，从而避免常驻一个完整 FP32 Key 副本。

## 8. 步骤四：SIMD 后端计算 Score

### 8.1 多核分工

Score 使用 omp for schedule(static) 分配 Task。不同任务拥有不重叠的 Query/Key 组合，因此每个 scores[b,i,j] 只有一个写入者。

线程之间不对同一个得分做 atomic add，不需要跨线程归约 head 的结果。一个得分的所有 head 都在同一个任务内合并。

### 8.2 单个任务中的循环结构

等价伪代码如下：

~~~python
for j in key_tile:
    key_fp32 = decode_bf16(Key[PA_address(b, j), :])
    for i in query_tile:
        if j >= visible(b, i):
            continue
        score = float32(0)
        for head_group in groups_of_four_heads:
            # 沿 D 做 SIMD 点积，4个独立累加器
            dot0, dot1, dot2, dot3 = dot_four_heads(Q, key_fp32)
            score += W0 * relu(dot0)
            score += W1 * relu(dot1)
            score += W2 * relu(dot2)
            score += W3 * relu(dot3)
        scores[b, i, j] = score
~~~

“四个 head 一组”指一个工作线程内的四个独立 FP32 累加器，不是四个 CPU 核。

### 8.3 Key 复用

一条 Key 向量只转换到当前线程私有的 kv[D]，随后供该任务中多个 Query 和全部 head 使用。D 循环中同一个 Key 分量同时参与四个 head 的点积。

query_tile>1 可使同一 Key 转换结果被多个 Query 复用；不同 Query 分块之间仍可能再次读取和转换相同 Key。提高 query_tile 可能减少重复转换，也可能减少任务数并增加工作集，需要实测。

### 8.4 SIMD 的实际含义

源码使用 omp simd reduction，要求编译器对 D 循环向量化。具体生成 SVE、NEON 还是标量代码，应检查目标机编译结果，不能仅凭 pragma 认定一定使用某条指令。

HISTSEG 路径对 SVE2 和 VL=256 有显式要求；Score 的 SIMD 指令选择由编译器和编译选项决定。

### 输出

所有可见位置的 FP32 最终 score；被 mask 和 padding 的位置保留 -∞。有效位置若产生非有限结果，则报错。

## 9. 可选步骤：BLAS 后端计算 Score

选择 score_backend=blas 时，必须先使用 scripts/build_cpu.sh --blas 编入 CBLAS 支持。

对于一个 Task，定义：

~~~text
m = 当前 Query 数 × H
n = 当前 Key 分块长度
A = FP32 Q，形状 [m,D]
B = 本线程打包的 FP32 Key，形状 [n,D]
C = A × Bᵀ，形状 [m,n]
~~~

实际调用单精度 cblas_sgemm，随后由本线程执行 FP32 ReLU、乘 W 和跨 head 求和。

- 该 Key tile 打包一次，供 m 条 Query/head 行使用。
- 不把整个 Key 按 head 广播复制。
- dots 临时矩阵只属于一个 Task 的局部 tile，不是完整 [B,S1,H,T] 中间矩阵。
- 因果 mask 在 GEMM 后的聚合阶段处理，部分本来不可见的点积可能已经被 GEMM 计算；这与 SIMD 后端提前跳过对应 Query/Key 的行为有区别。
- 外层 OpenMP 使用多线程，BLAS 内部应设为单线程，避免 P 个 OpenMP 线程各自再启动一组 BLAS 线程。
- 扫描脚本设置 OPENBLAS_NUM_THREADS、MKL_NUM_THREADS、BLIS_NUM_THREADS 为 1；CPUOp 还尝试调用可用的库线程设置函数。实际 BLAS 是否遵守应结合具体库确认。

BLAS 并非默认一定更快。tile 较小时，打包及函数调用成本可能较高，需要在相同核数和输出要求下实测。

## 10. 步骤五：将 FP32 得分编码为 UINT32 排序键

### 输入与输出

输入：一个有限 FP32 score。  
输出：一个 UINT32 保序键 key。

实现逻辑：

~~~cpp
if (score == 0.0f) score = 0.0f;  // 统一 -0 与 +0
uint32_t bits;
std::memcpy(&bits, &score, 4);
uint32_t key =
    bits ^ ((bits >> 31) ? 0xffffffffu : 0x80000000u);
~~~

这不是把小数强制转换成整数，而是读取 IEEE FP32 位模式后进行符号相关变换。

| 得分 | FP32 位模式 | 保序键 |
|---:|---|---|
| -2 | 0xC0000000 | 0x3FFFFFFF |
| 0 | 0x00000000 | 0x80000000 |
| 1 | 0x3F800000 | 0xBF800000 |
| 2 | 0x40000000 | 0xC0000000 |

普通有限值的分数越大，保序键越大。小数信息没有被截断，正负零按同分处理。

非有限的有效得分已经在 Score 阶段拒绝。mask 槽位中的 -∞ 不进入有效候选区间，不参与这一步筛选。

## 11. 步骤六：四轮 radix 查找第 K 大阈值

### 11.1 按字节由高到低缩小候选

UINT32 排序键共有 4 个字节，依次处理 byte=3、2、1、0。每轮对当前候选的目标字节建立 256 桶直方图。

设 hist[z] 是字节值 z 的候选个数，rank 是仍需在当前候选中寻找的名次。初始 rank=K。从桶 255 向 0 扫描：

~~~python
for z in range(255, -1, -1):
    if rank > hist[z]:
        rank -= hist[z]
    else:
        target = z
        break
~~~

将 target 拼接进阈值前缀，只留下目标桶中的候选，再处理下一个较低字节。

### 11.2 示例

某轮各高位桶的数量为：

| 桶 | 数量 |
|---:|---:|
| 250 | 2 |
| 249 | 3 |
| 248 | 4 |

若 rank=4，则先跳过桶 250 的 2 个元素，rank 变为 2；目标位于桶 249。下一轮只需要在桶 249 的 3 个候选中找第 2 大。

高位桶中已确定更大的元素不会丢失最终 TopK 资格：阈值确定后会重新扫描原始 Score，将所有大于阈值的元素收集回来。

### 11.3 输出

四轮后得到精确的 UINT32 第 K 大阈值。这里不是按桶做近似 TopK，也不是只比较 FP32 的高 8 位或高 16 位。

当前代码执行完整四轮；没有在候选只剩一个时提前退出的优化。每轮的候选压缩在本线程 keys 数组内完成，不改变原始 Score 和逻辑索引顺序。

## 12. HISTSEG 如何形成 256 桶直方图

### 12.1 单条指令统计的是 128-bit segment 内的匹配次数

当前实现调用：

~~~cpp
svhistseg_u8(queries, data)
~~~

对于每个查询 lane，它统计对应查询字节在该 lane 所属的 128-bit data segment 中出现的次数。

SVE VL=256 bit，即 32 字节，包含两个 128-bit segment，每段 16 个 UINT8 元素。单条 HISTSEG 不是直接产生完整的全局 256 桶直方图。

### 12.2 本项目的查询向量

对 base=0，查询向量为：

~~~text
第一个 segment：[0,1,2,...,15]
第二个 segment：[0,1,2,...,15]
~~~

对 base=16，则两段均变成 [16,17,...,31]；如此处理到 base=240。

因此总共 16 组查询，覆盖全部 256 个桶。

### 12.3 具体流程

1. 从 UINT32 候选键中提取本轮字节，存入线程私有 scratch，类型为 UINT8。
2. 固定一个 base，每次加载 32 个数据字节。
3. HISTSEG 分别计算两个 segment 对这 16 个桶的计数。
4. 将 UINT8 计数逐级展开到 UINT16、UINT32。
5. 对应桶的两个 segment 计数相加，跨数据块累加到 UINT32 寄存器。
6. 数据扫描结束后，将这组 16 个桶写回 hist。
7. 切换下一个 base，直到填满 hist[256]。

源码将桶组循环放在外层、数据块循环放在内层，使累加值在寄存器中保留。代价是当前字节 scratch 会被扫描 16 遍。

两个 segment 中单个桶的计数最多各为 16；展开到 UINT32 后再长期累加，避免使用 UINT8 累加器溢出。

### 12.4 尾部不足 32 字节

加载使用谓词，越界 lane 填零；但 HISTSEG 本身不带对应的有效 lane 谓词，因此填充零仍会贡献到桶 0。

代码最终修正：

~~~cpp
h[0] -= (32 - n % 32) % 32;
~~~

例如 n=35，最后加载只有 3 个有效字节，其余 29 个 lane 补零，最终从桶 0 减去 29。真实数据中的零仍被保留统计。

### 12.5 约束与对照

- hist=histseg 时，每个工作线程都检查 VL=256；不满足时直接报错。
- 没有 SVE2 的构建不能静默代替为 HISTSEG 路径。
- hist=scalar 使用逐元素提取字节并递增计数，作为对照；不改变 radix 阈值算法。
- SVE 的向量宽度限制专门针对当前 HISTSEG 实现，不能因为源码使用 SVE 就认为它已经支持所有 VL。

## 13. 步骤七：GT/EQ 筛选，形成最终 K 个索引

令阈值为 kth_key：

1. 扫描原始有效 Score，收集 key>kth_key 的全部索引。
2. 再收集 key==kth_key 的索引，补足 K 个。
3. 边界同分时优先选择较小的逻辑索引。

例如 score=[7,2,9,8,1]，K=4：

~~~text
阈值：2
GT 索引：[0,2,3]
EQ 补充：[1]
unordered 输出：[0,2,3,1]
对应得分：[7,9,8,2]
~~~

输出集合是正确的 TopK，但没有按得分降序排列。

### 有效元素不足 K

- v=0：整行输出保持 -1。
- 0<v≤K：所有有效索引 0～v-1 均被选中，跳过 radix 查找，剩余槽位为 -1。
- v>K：执行四轮 radix 和 GT/EQ 筛选。
- padding Query 不建立有效 Row，输出保持 -1。

### 可选最终排序

sorted 模式只对已经选中的 min(v,K) 个索引追加 std::sort，比较原始 FP32 Score：先分数降序，再同分索引升序。

unordered 模式跳过这一操作。它仍确定 TopK 集合及边界同分选择，不代表随机输出。

当前 CPU 算法做全行 radix 筛选，没有复现 NPU 的 trunk 逐块候选合并。即使集合相同，也不能保证与 NPU 输出序列逐元素相同；NPU 同分规则也需单独核实。

## 14. TopK 并行原理：行间并行与行内并行

### 14.1 自动分组

默认 groups=0：

$$
G=\max(1,\min(P,R))
$$

P 在执行中以实际 OpenMP 线程数为准；如果实际线程数与请求数不一致，函数会返回错误，而不是悄悄按较少线程报告结果。

显式指定 groups>0 时，采用指定组数并限制在实际线程数范围内。改变 groups 只影响 TopK 阶段，Score 仍由全部工作线程处理 Task。

### 14.2 路径 A：G=P，每个线程独立处理整行

当有效 Query 行足够多，或显式选择 G=P 时，使用：

~~~cpp
#pragma omp for schedule(dynamic,1)
for (each row)
    select_row(...);
~~~

每次由一个线程负责该行完整 radix、GT/EQ 和可选排序。不同线程处理不同的行，没有 radix 每一轮的跨线程直方图汇总和全局 barrier。

当 R>P 时，多行按动态调度交给线程处理，有助于平衡各行有效长度不同造成的差异。代价是单行不能同时使用多个核。

### 14.3 路径 B：G<P，一行由一个线程组合作

把 P 个线程分成 G 个连续线程组。组 g 的线程边界：

$$
first_g=\lfloor gP/G\rfloor,\quad
last_g=\lfloor(g+1)P/G\rfloor
$$

组大小可以不相等。例如 P=7、G=3，组大小为 2、2、3。

一个组在一轮 wave 中负责一条 Row。组内第 local 个线程，在 count 个组线程中负责：

$$
begin=\lfloor v\cdot local/count\rfloor
$$

$$
end=\lfloor v\cdot(local+1)/count\rfloor
$$

各线程区间连续、不重叠、合起来覆盖整行。

### 14.4 每轮 radix 的线程协作

1. 每线程对自己的候选区间计算私有 hist[256]。
2. barrier 等待所有线程完成直方图。
3. 组长将本组各线程直方图按桶相加。
4. 组长计算目标桶、新 rank 和阈值前缀。
5. barrier 保证组员看到更新。
6. 各线程在自己的 keys 中只保留目标桶候选。
7. 下一轮的同步防止组长过早覆盖共享状态。

这里没有多个线程对同一 hist 桶做原子递增。桶汇总由组长串行完成，开销与组大小及 256 个桶相关。

### 14.5 并行收集输出，为什么不需要原子抢位置

阈值确定后，每线程统计：

~~~text
gt_t：本区间内 key>阈值 的数量
eq_t：本区间内 key==阈值 的数量
~~~

组长按逻辑区间顺序计算前缀和：

~~~text
offg_t = 前面线程的 GT 数之和
offe_t = 前面线程的 EQ 数之和
ng = 全组 GT 总数
need = K - ng
~~~

线程 t：

- 从输出位置 offg_t 开始写自己的 GT 索引。
- EQ 的全局序号从 offe_t 开始，仅当该序号小于 need 时，写到 ng+EQ序号。

各线程的写入位置互不重叠，避免 atomic fetch_add。线程分区对应递增索引，区间内又按索引递增扫描，因此边界同分时能优先选择较小索引。

### 14.6 当前实现的同步限制

路径 B 使用的是整个 OpenMP team 的 barrier，不是组内独立 barrier。多个组按 wave 推进，快组可能等待慢组；即使某组没有有效行，仍需参与全局同步。

因此它有组间同步成本，不是完全独立的多个子线程池。线程数很多、行很短或组间行长差异很大时，这会限制扩展性。

## 15. 三个典型并行例子

### 15.1 decode_64k：R=1，P=48

- Score：512 个 Key tile 任务由 48 个线程静态分工。
- TopK：自动 G=1，一行由全部 48 个线程合作。
- 每线程首先处理约 65536/48 个有效分数。
- 各自建立直方图，由一个组长汇总，执行四轮筛选。
- 这种行内并行避免单 Query 的 TopK 只能使用一个核，但直方图汇总和 barrier 成本会随线程数增加。

### 15.2 原 A2 回归形状：B=18，S1=3，全部 Query 有效

R=54。若 P=48，自动 G=48，走独立整行 TopK 路径，54 行动态分配给 48 个线程；不是仅按 18 个 Batch 分配线程。

若 P=96，自动 G=54，走多线程组路径，每组约 1～2 个线程，并存在全局 barrier。开启 SMT 后，算法路径和同步成本也可能变化，不能只把结果解释为“线程数翻倍”。

### 15.3 P=12，R=3

自动 G=3，每组 4 个线程并行处理一行。每个组只汇总自己的直方图，但源码中的 barrier 会等待全部 12 个线程。

## 16. 工作区、数据共享与内存开销

| 数据 | 是否共享 | 生命周期及写入规则 |
|---|---|---|
| BF16 输入、PA 表、有效长度 | 全线程只读 | 调用前已驻留 |
| FP32 Q/W | 共享 | prepare 并行写不同位置，随后只读 |
| FP32 Score | 共享 | 每个有效位置由一个 Score Task 写 |
| indices | 共享 | 不同行或前缀和分配的不同槽位写入 |
| Worker.hist/keys/bytes/kv | 每线程私有 | 重复调用复用 |
| packed_key/dots | 每线程私有，可选 BLAS 构建 | Task 局部使用、工作区复用 |
| Group 状态 | 每组共享 | 组长更新，barrier 后组员读取 |

每线程 keys 容量为 T 个 UINT32，bytes 容量为 T 个 UINT8，因此仅这两项约为：

$$
M_{\mathrm{keys+bytes}}=5PT\ \text{字节}
$$

decode_64k：

| 工作线程数 | keys+bytes 总容量 |
|---:|---:|
| 1 | 320 KiB |
| 48 | 15 MiB |
| 96 | 30 MiB |

这是当前容量分配策略，即使某线程只处理一小段候选，也预留完整 T 容量，不能误认为已经按分区最小化分配。

其他主要内存：

~~~text
FP32 Q：4×B×S1×H×D
FP32 W：4×B×S1×H
FP32 Score：4×B×S1×T
INT32 indices：4×B×S1×K
SIMD Key 临时区：约4×P×D
BLAS packed_key：约4×P×tile×D
BLAS dots：约4×P×query_tile×H×tile
~~~

BLAS 临时区在编入 BLAS 支持时分配，不仅在选择 blas 后端时才有容量。以上未计 Row/Task、直方图、对象和分配器开销，以及输入文件加载所占内存。

每线程/每组结构体使用 alignas(64)；这指定结构体对齐，不等于内部 std::vector 的堆数据都按 64 字节对齐，也不能据此保证完全消除 false sharing。

## 17. NUMA、物理核与 SMT

scan_cpu.py 为每档核数启动新进程。topology.py 从 Linux sysfs 的 thread_siblings_list、在线 CPU 集合和进程可用 affinity 中识别物理核。

以用户服务器 node0 的 CPU 0～95、相邻 SMT2 为例：

| 模式 | 4 个物理核示例 | 工作线程数 |
|---|---|---:|
| 不加 --smt | 0、2、4、6 | 4 |
| 加 --smt | 0、1、2、3、4、5、6、7 | 8 |

不加 --smt 只是本测试每物理核使用一个线程，不会在系统中关闭 SMT，也不能阻止其他进程占用兄弟线程。

扫描器设置：

- numactl --physcpubind：限制进程可使用的逻辑 CPU。
- numactl --membind：绑定内存分配到指定 NUMA 节点。
- OMP_PLACES：每个工作线程对应一个逻辑 CPU。
- OMP_PROC_BIND=TRUE：禁止工作线程随意迁移。
- OMP_DYNAMIC=FALSE、OMP_THREAD_LIMIT：避免线程数自动变化。
- OMP_MAX_ACTIVE_LEVELS=1，以及 BLAS 线程设置：限制嵌套并行。
- 默认 OMP_WAIT_POLICY=PASSIVE；可测试 ACTIVE，但它是等待策略提示，不是性能提升保证。

运行后验证实际线程数、CPU 集合和 VL。当前扫描入口主要评估单 NUMA 节点，不是已经实现跨四个 NUMA 节点的数据分片和局部归约方案。

## 18. 计时区间和输入输出边界

| 字段 | 对应区间 | 包含内容 |
|---|---|---|
| prepare | t0→t1 | Row/Task 构造、进入 OpenMP、Q/W 转换、Score/输出初始化及相关同步 |
| score | t1→t2 | Key 寻址/读取/转换、SIMD 或 BLAS 点积、ReLU、加权求和及相关同步 |
| topk | t2→t3 | 编码、直方图、阈值、候选收集、可选最终排序及相关同步 |
| total / native | t0→退出 OpenMP 后 | 完整原生算法计时 |
| host | Python 调用前→返回后 | 原生执行、ctypes/Python 封装；per-call 时还包括对象构造/释放 |

t0 位于原生参数和 PA 合法性检查之后，不应把 native 描述为所有接口检查都包含的严格函数入口时间。

默认复用模式中，初始工作区分配、文件读取、哈希验证、正确性验证、保存 indices/scores 均在主计时外。Q/W 转换、Key 局部转换、Score 初始化仍在每次原生计时内，没有被移到预处理外。

native_median_us 和 stage_median_us.total 来自同一组样本，不应相加。阶段分别取中位数，不能要求阶段中位数之和等于总中位数；total 另外包含并行区退出等尾部开销。

CPU 是目标服务器实测，不是 CA 仿真。CPU 原生时间可与已核实范围和时间基准的完整 NPU LI 模型时延比较，但不能与仿真软件的 Host 墙钟直接比较。当前尚未确认的 CA Total tick 不应在本文件中当作已测得的 A5 时延。

## 19. 正确性验证与输出契约

verify=golden 在计时外执行两层检查：

1. 原生 Score 上的选择检查：独立使用 NumPy lexsort 建立参考，检查输出数量、范围、重复、TopK 阈值和排序模式。
2. 独立 FP32 Score：NumPy 矩阵乘后 ReLU、乘 W、跨 head 求和，与原生得分按容差比较。

unordered 的阈值验收允许第 K 大附近同分时存在多个合法集合，同时记录 exact_set_mismatch_rows 作为诊断；这个字段非零不一定导致 unordered 验收失败。

本原生实现自身有确定的较小索引优先规则，但不应把 unordered 阈值检查通过解读为已验证 NPU 的全部排序与同分契约。

验证代码为了建立参考可能执行完整排序；它在计时区间外，因此不属于 unordered 原生 TopK 的性能开销。

## 20. 运行与参数选择

### 20.1 单次配置

~~~bash
python3 generate_case.py --preset decode_64k --out cases/decode_64k

python3 cpu_bench.py --case cases/decode_64k \
  --threads 8 --tile 128 --query-tile 1 \
  --hist histseg --score-backend simd --order unordered \
  --warmup 10 --iters 100 --verify golden --save-scores \
  --out results/cpu_decode_64k_8threads
~~~

直接 cpu_bench 不自动按物理核选择 CPU，也不自动绑定 NUMA；正式扩展性测试使用扫描器。

### 20.2 单 NUMA 物理核扫描

~~~bash
python3 scan_cpu.py --case cases/decode_64k \
  --node 0 --cores 1,2,4,8,16,24,32,48 \
  --tile 128 --query-tile 1 \
  --hist histseg --score-backend simd --order unordered \
  --warmup 10 --iters 100 --verify golden --save-scores \
  --out results/cpu_decode_64k_scan
~~~

增加 --smt 并换一个输出目录，可比较同一物理核数下 SMT2 的影响。--cores 是依次测试的物理核数量，不是 CPU ID 列表。

### 20.3 参数与算法部分的对应关系

| 参数 | 改变的行为 |
|---|---|
| --threads | 原生 OpenMP 工作线程数；scan_cpu 自动设置 |
| --tile | Score Task 的 Key 分块大小，不改变 PA block_size |
| --query-tile | Score Task 的 Query 分块大小与 Key 复用范围 |
| --groups | TopK 分组；0 自动，正数显式指定 |
| --hist | HISTSEG 或标量直方图；不改变四轮 radix 逻辑 |
| --score-backend | SIMD 点积或 CBLAS SGEMM |
| --order | 是否对选中的 K 项追加分数排序 |
| --input-copies | 在多份相同输入数据副本之间轮换；不保证冷缓存 |
| --allocation | 工作区复用或每次构造/释放；主要影响 host 时间 |
| --warmup / --iters | 不计入统计的预热次数 / 正式采样次数 |
| --verify | none、selection、golden，均在计时外执行 |
| --save-scores | 将最终一次 FP32 Score 保存为 scores.f32 |
| --node / --cores / --smt | 扫描器的 NUMA、物理核数和兄弟线程选择 |
| --dry-run | 扫描器只打印绑核与命令，不执行测量 |

即使已经将 cpu_bench.py 的函数与 argparse 默认值都改为 unordered，显式写出 --order unordered 仍便于记录实验条件；它不是额外的优化开关。

### 20.4 查看结果

~~~bash
cat results/cpu_decode_64k_scan/summary.csv
python3 -m json.tool results/cpu_decode_64k_scan/cores_8/result.json
~~~

以首个扫描点 N0 为基准：

$$
speedup(N)=T(N_0)/T(N)
$$

$$
efficiency(N)=speedup(N)/(N/N_0)
$$

当首点为 1 核，就是常见的 T1/TN 和 T1/(N×TN)。开启 SMT 时这里的 N 仍是物理核数，应另外观察 threads 列。

输出目录应使用新的或空目录；已有输入和单点结果目录默认拒绝覆盖。

## 21. 当前优化、局限与扩展性解释

### 已实现的优化

- Score 沿 Query/Key 分块并行，不受 Batch 数限制。
- 一次调用只有一个外层 OpenMP 并行区。
- Key 局部转换并复用，不按 head 复制完整 Key。
- SIMD 四 head 共享 Key 分量，可选局部 SGEMM。
- TopK 根据行数切换行间和行内并行。
- 线程私有直方图、候选数组和 Key 缓冲区。
- 输出位置通过计数前缀和确定，不逐元素竞争原子计数器。
- unordered 省去最终 K 项排序。
- v≤K 时跳过 radix 筛选。
- 工作区、输出及中间缓冲区复用。

### 尚存在的成本

- 完整 FP32 Score 需要写入再读取，暂未融合 Score 与 TopK 流式处理。
- prepare 每次初始化完整 [B,S1,T] Score，即使部分位置被 mask。
- 每线程候选容量按完整 T 分配，线程数大时占用较多内存。
- HISTSEG 每个 radix 轮次要对字节 scratch 做 16 组扫描。
- 多线程组路径使用全局 barrier，组长串行汇总直方图。
- sorted 模式的最终 K 项 std::sort 在每行由单个线程执行。
- 重复访问一份输入属于热缓存倾向的测试条件，轮换缓冲区也不等于严格冷缓存。
- 增加 CPU 核数不保证线性加速；带宽、同步、任务颗粒度、缓存、频率及 SMT 资源共享都会影响结果。

这些是解释扫描曲线的因素，不是目标机性能结论。应分别观察 prepare、score、topk 及 total，再判断哪个阶段已经饱和。

## 22. 文件到算法步骤的索引

| 文件 / 函数 | 主要职责 |
|---|---|
| caseio.py / Case | 输入精度、布局、长度、PA、哈希校验 |
| generate_case.py / generate | 代表形状、BF16 输入、PA 映射与共享文件生成 |
| cpu_api.py / CPUOp | 工作区建立、ctypes 调用、数组复用与线程使用保护 |
| native/cpu_li.cpp / li_execute | 任务构造、并行准备、Score、TopK 及原生计时 |
| native/cpu_li.cpp / select_row | 独立整行 TopK 路径 |
| native/cpu_li.cpp / Worker、Group、Workspace | 私有及共享工作区、分组状态 |
| native/histogram.hpp / float_to_sortable_key | FP32 保序 UINT32 编码 |
| native/histogram.hpp / histogram_sve2 | VL=256 的 HISTSEG 256 桶统计 |
| native/histogram.hpp / histogram_scalar | 标量直方图对照 |
| topology.py / discover | 实际物理核/SMT 拓扑识别 |
| scan_cpu.py / main | 独立进程核数扫描、绑核、NUMA 绑定和汇总 |
| cpu_bench.py / run_benchmark | 预热、采样、统计、验证及结果保存 |
| validation.py / golden_scores | 独立 FP32 Score 参考 |
| validation.py / selection_on_scores | 输出集合、阈值及排序诊断 |

本文依据上述 CPU 源码进行说明，不声称已完成目标 ARM 服务器性能验证，也不将 CA 模型估计等同于 A5 上板测量。

## 附录：本次阅读的核心源码指纹

以下 SHA256 用于标识文档审阅时的本地代码，不代替目标机库二进制的哈希。

| 文件 | SHA256 |
|---|---|
| native/cpu_li.cpp | 74cdeab8e1e1e54dab4d29be858dedba31c1ba00466872233d262a9920b7d7db |
| native/histogram.hpp | 4527cb597468d399b58c93521c0cdd8df67c7cacca05e5f17083c4bdaf5495ab |
| cpu_api.py | 6e072da7d9db81ab393564a5e49cf8fe82db3c97b5b5e08b03a8cec02bff1d32 |
| caseio.py | 4132e3b78b785b46ca735c0b4c33c294431c418b9d1a4e02c1b679532cc63290 |
| scan_cpu.py | 6ac06bf51a2cda4c7f930fc8c2081aa102744c7cba3726a47b7c2e2d195fbf14 |
| topology.py | a91b8b76cc0b3ce34d0357a7de09e6997c0bf8bd071bb635a87d90805c84cf48 |
| validation.py | 2bcbcc71bfad283d430f4894fa4d56ac4dcaf51ba95af57b61a46eec07b2a704 |

更新于 2026-11-09。
