# RotaryPosEmb 接口与实现说明

本文介绍本仓库的 Ascend C Kernel 直调工程。题目入口：[RotaryPosEmb](https://cannjudge.cn/public/op_challenge_jingdong_prelim/rotaryposemb)。

## 功能与计算

对每个 token 和 attention head 的最后一维执行旋转位置编码。设最后一维为偶数 `D`，`i = 0, …, D/2-1`，位置为 `p`：

- 逆频率：`inv_freq[i] = theta^(-2i/D)`。
- 角度：`angle[i] = p * inv_freq[i]`；将角度复制为两半后计算正弦、余弦。
- 将输入最后一维分为 `a`、`b` 两半，`rotate_half(x) = [-b, a]`。
- 输出：`y = x * cos(angle) + rotate_half(x) * sin(angle)`。

NumPy 参考计算见 [RotaryPosEmb.py](../scripts/RotaryPosEmb.py)：参考计算先转为 float32，再转换回输出类型。具体数值以实际输入和实现为准。

## 参数

| 参数 | 方向 | 示例形状／类型 | 说明 |
| --- | --- | --- | --- |
| `x` | 输入 | `(B, S, H, D)`；float16 或 float32 | 连续排布的输入，最后一维为旋转维度。 |
| `positions` | 输入 | `(B, S)`；int32 | 每个 token 对应的位置，同一 token 的各 head 共用。 |
| `theta` | 属性 | float；本地示例为 10000 | 逆频率计算的基数。 |
| `y` | 输出 | 与 `x` 相同的形状和类型 | 写入调用者提供的设备内存。 |

本工程的参考计算按最后一维均分；调用方应使用正的偶数 `D`、正的 `theta`，并保证形状、数据类型、设备地址与实际分配一致。具体赛题输入范围以[题目页面](https://cannjudge.cn/public/op_challenge_jingdong_prelim/rotaryposemb)为准。源码入口按 `x` 的 dtype 选择 half 或 float 分支，调用方需保证只传入约定的类型；不要依赖入口替调用方完成参数检查。

## Kernel 直调接口

定义见 [kernel.asc](../kernel.asc)，示例调用见 [main.asc](../main.asc)。其中 `TensorInfo` / `TensorGroupInfo` 由调用侧提供，地址参数指向设备内存：

```cpp
extern "C" void run_kernel(
    GM_ADDR x, const TensorGroupInfo& info_x,
    GM_ADDR positions, const TensorGroupInfo& info_positions,
    GM_ADDR y, const TensorGroupInfo& info_y,
    int64_t availableCoreNum, aclrtStream stream, float theta);
```

`main.asc` 演示初始化 ACL、选择设备 0、查询可用 Vector 核数、创建 stream、分配与复制输入，调用 `run_kernel`，再同步 stream 和拷回结果。入口将维度和核数用于选择实现路径并通过 `<<<...>>>` 启动内核。调用方负责设备上下文、内存、元数据生命周期、同步和错误处理；该工程不提供 ACLNN 注册接口。

## 实现概览

- Host 侧根据输入形状、元素类型与可用核数计算任务规模并选择 Kernel 分支。
- Kernel 按 token、head 或 tile 划分工作，完成位置与频率相关的计算和旋转；不同规模使用各自的数据搬运与计算路径。
- 原始提交中的 `kernel.asc` 和 `main.asc` 保持原样；阅读时可从 `run_kernel` 的分派开始，再查看对应 Kernel 的计算与搬运实现。

## 构建与本地示例

需要昇腾 NPU、支持该工程的 CANN Toolkit / Ascend C 编译环境、CMake ≥ 3.16、make、Python 3、NumPy、Bash 和 `timeout`。工程默认 `NPU_ARCH=dav-2201`；实际适配范围和可用 CANN 版本以对应设备环境为准。

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
bash run.sh
```

按实际安装路径加载 CANN 环境。`run.sh` 会删除并重建工程目录中的 `build/`，生成 case0 数据，运行设备程序并校验输出。示例固定使用 `(1, 7, 1, 16)` 的 float32 输入、`(1, 7)` 的 int32 位置和 `theta=10000`；要使用其他形状，需同步调整调用侧元数据、内存分配、数据生成与校验逻辑。

若手动构建需要指定架构，可在已加载 CANN 环境后运行 `cmake -S . -B build -DNPU_ARCH=dav-2201` 和 `cmake --build build`。架构标识和设备兼容性应自行确认。

## 结果校验

`run.sh` 在 `build/` 中调用 `scripts/verify_result.py 0`。case0 采用 float32，`rtol=1e-4`、`atol=1e-4`，逐元素比较参考数据，允许不匹配比例 `tol=1e-4`。校验器先检查两个文件的字节长度相同、长度符合数据类型对齐，并检查 case0 有 112 个输出元素；长度错误直接失败，不计入数值容差。非零退出状态表示运行或校验失败。

无需 NPU 的校验器测试：

```bash
python3 -m pip install numpy
python3 -m unittest discover -s tests -v
```

这些测试覆盖校验器本身，不代替昇腾设备上的算子验证。
