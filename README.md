# RotaryPosEmb：2026 CANN 挑战赛算子

本项目开源 [2026 华为 CANN 挑战赛 RotaryPosEmb](https://cannjudge.cn/public/op_challenge_jingdong_prelim/rotaryposemb) 的 Ascend C 实现。工程通过 **Kernel 直调**运行：`main.asc` 负责设备和数据准备，`kernel.asc` 提供算子入口及 Kernel。

对旋转位置编码（RoPE）的输入按位置和频率计算旋转角，并完成 `y = x * cos(angle) + rotate_half(x) * sin(angle)`。参数、公式和调用说明见 [RotaryPosEmb 接口文档](docs/RotaryPosEmb.md)。

## 参数概览

| 参数 | 说明 |
| --- | --- |
| `x` | 输入，`(B, S, H, D)`，float16 或 float32。 |
| `positions` | 位置索引，`(B, S)`，int32。 |
| `theta` | 频率基数，示例中为 10000。 |
| `y` | 输出，与 `x` 的形状和类型相同。 |

## 环境要求

需要匹配的昇腾 NPU、CANN Toolkit（含 Ascend C 编译环境）、CMake ≥ 3.16、make、Bash、`timeout`、Python 3 和 NumPy。工程默认编译目标为 `dav-2201`；实际适配设备和 CANN 版本以环境为准。`ml_dtypes` 是脚本中的可选依赖。

## 快速开始

```bash
git clone https://github.com/xinjiayuan-diyishenqing/cann-challenge-2026-rotaryposemb.git
cd cann-challenge-2026-rotaryposemb
source /usr/local/Ascend/ascend-toolkit/set_env.sh
bash run.sh
```

CANN 路径请按实际安装位置调整。脚本会删除并重建工程目录下的 `build/`，生成数据、运行设备程序并校验 case0。示例使用 `(1, 7, 1, 16)` 的 float32 输入和 `theta=10000`；它只运行这一个本地样例。改变输入形状时，需要同时调整调用侧的元数据、内存、数据生成和校验。

校验脚本自身的测试不需要 NPU：

```bash
python3 -m pip install numpy
python3 -m unittest discover -s tests -v
```

## 目录结构

```text
├── kernel.asc                  # Ascend C 算子和 Kernel 直调入口
├── main.asc                    # ACL 初始化、设备内存和调用示例
├── CMakeLists.txt              # 构建配置
├── run.sh                      # 构建、生成 case0 并运行校验
├── data_utils.h                # 本地文件读写辅助代码
├── scripts/
│   ├── RotaryPosEmb.py         # NumPy 参考计算
│   ├── gen_data.py             # case0 数据生成
│   └── verify_result.py        # 输出文件和数值校验
├── tests/
│   └── test_verify_result.py   # 校验脚本边界测试
├── docs/
│   └── RotaryPosEmb.md         # 参数、接口与实现说明
├── .github/workflows/          # 校验脚本自动测试
└── LICENSE
```

`kernel.asc` 和 `main.asc` 保留最终提交的直调实现。仓库中的校验脚本和文档在公开整理时单独维护。

## 许可与来源

本仓库原创算子代码采用 [MIT License](LICENSE)。工程中的赛题构建和验证脚手架随提交工程保留，其原有权利及适用许可应以赛事提供方的规定为准；本仓库许可不授予第三方未授权的权利。
