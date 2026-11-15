# RotaryPosEmb：2026 CANN 挑战赛算子

本项目保存 2026 华为 CANN 挑战赛 [RotaryPosEmb 题目](https://cannjudge.cn/public/op_challenge_jingdong_prelim/rotaryposemb)的最终提交工程，使用 Ascend C 实现，采用 **Kernel 直调**方式运行。

## 功能

对输入的旋转位置编码维度执行 RoPE。参考实现见 [scripts/RotaryPosEmb.py](scripts/RotaryPosEmb.py)：输入 `x` 的形状为 `(B, S, H, rope_dim)`，位置索引 `positions` 的形状为 `(B, S)`、类型为 int32，`theta` 为频率基数；输出与输入的形状和数据类型相同。

参考计算为 `y = x * cos(angle) + rotate_half(x) * sin(angle)`，其中角度由位置和逆频率计算，`rotate_half` 将最后一维的两半变换为 `[-x₂, x₁]`。实际支持的输入组合和约束应以赛题与源码为准。

## 工程结构

- [kernel.asc](kernel.asc)：算子实现和直调入口，包含多核任务划分与分支实现。
- [main.asc](main.asc)：ACL 设备、内存和流管理，以及直调示例。
- [CMakeLists.txt](CMakeLists.txt)、[run.sh](run.sh)：构建与本地示例流程。
- [scripts/](scripts/)：输入数据生成、NumPy 参考计算和输出校验。

## 构建和运行

需要匹配的昇腾 NPU、CANN Toolkit（含 Ascend C 编译环境）、CMake 3.16 或以上、Python 3 和 NumPy。工程默认编译目标为 `dav-2201`；实际兼容性取决于设备与 CANN 版本。`ml_dtypes` 在数据生成脚本中为可选依赖。

在昇腾环境中执行：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
bash run.sh
```

若 Toolkit 不在默认路径，请先设置 `ASCEND_HOME_PATH` 并加载相应环境。脚本会删除并重建工程下的 `build/`，生成测试数据，运行并校验 **case0**；本地示例并不覆盖全部评测用例。`main.asc` 中示例形状为 `(1, 7, 1, 16)`，`theta=10000`。如需运行其他输入，需相应调整示例入口及数据。

## 许可与来源

本仓库原创算子代码采用 [MIT License](LICENSE)。工程中的赛题构建和验证脚手架随提交工程保留，其原有权利及适用许可应以赛事提供方的规定为准；本仓库许可不授予第三方未授权的权利。
