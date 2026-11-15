# P2 构建与最小样例（MNIST）

> 记录时间：2026-11-15 15:48–16:00 (GMT+8)
> 结论：**上游最小样例的完整链路已跑通——真实 KZG 证明生成 + 验证成功，非 Mock。**

## 1. 代码基线

| 项目 | 值 |
| --- | --- |
| 克隆来源 | `https://github.com/patrick-andstar/zkml2.git`（本轮已把基线推送到个人仓库） |
| 检出提交 | `85ee9a5`（`docs: add reproduction plan and P1 environment inventory`） |
| 上游基线提交 | `50d0cf3`（`clean some bench files`，来自 `uiuc-kang-lab/zkml`） |
| 服务器路径 | `/root/autodl-tmp/zkml2` |
| remotes | `mine` = 个人仓库；`upstream` = `uiuc-kang-lab/zkml` |
| 本地未跟踪改动 | 无（工作区仅 `.workbuddy/`，未入库） |

## 2. Rust 工具链

| 项目 | 值 |
| --- | --- |
| 安装方式 | `sh.rustup.rs`，`--default-toolchain nightly --profile minimal` |
| 分发源 | 清华镜像 `mirrors.tuna.tsinghua.edu.cn/rustup`（`static.rust-lang.org` 也可达，镜像更快） |
| 实际版本 | `rustc 1.102.0-nightly (8eb778d04 2026-11-12)` / `cargo 1.102.0-nightly (ede6cefea 2026-11-11)` |
| `RUSTUP_HOME` / `CARGO_HOME` | `/root/.rustup` / `/root/.cargo` |
| `CARGO_TARGET_DIR` | `/root/autodl-tmp/zkml2/target`（指到数据盘，避免撑爆 30 G 系统盘） |

### 兼容性：无需任何补丁

`halo2/` 是 vendored 的 2023 年代码（`halo2_proofs 0.2.0`，其 `rust-toolchain` 钉的是 `1.64.0`），本轮用 **2026-11-12 的 nightly** 直接编译成功，跨约 3 年。

```
Finished `release` profile [optimized] target(s) in 5m 15s
exit=0
```

仅有 warning（`dead_code`、`unused_dependencies`），无 error，**未做任何源码修改**。

> 说明：根目录无 `rust-toolchain`，`halo2/rust-toolchain`（1.64.0）只在 `cd halo2` 时生效，不影响根构建。为防歧义，构建时显式设了 `RUSTUP_TOOLCHAIN=nightly-x86_64-unknown-linux-gnu`。

### 产物

```
estimate_cost  estimate_size  test_circuit  time_circuit  verify_circuit  verify_wav
```

## 3. MNIST 样例实测

运行目录 `/root/autodl-tmp/runs/p2_mnist`（独立目录，`params_kzg/` 与 `params_ipa/` 就地创建，不污染仓库）。

### 3.1 MockProver（仅约束检查）

```bash
test_circuit model_dir/model.msgpack model_dir/inp.msgpack
```

- 输出含 `Proof verified!`
- wall 0.724 s，峰值 RSS **0.109 GiB**
- ⚠️ 这一步**没有密码学证明**，只是约束满足检查。

### 3.2 真实 KZG 证明 + 验证

```bash
time_circuit model_dir/model.msgpack model_dir/inp.msgpack kzg
```

| 指标 | 值 |
| --- | --- |
| Proving time | **8.269 s** |
| Verifying time | **16.83 ms** |
| proof 大小 | **3360 B** |
| proof verified | ✅（`verify_kzg` 内 `assert!`，未 panic） |
| 端到端 wall（含参数生成、keygen） | 13.277 s |
| 峰值 RSS | **0.241 GiB** |

产物（运行目录）：

| 文件 | 大小 |
| --- | --- |
| `params_kzg/15.params` | 4,194,564 B |
| `pkey` | 36,725,924 B |
| `vkey` | 25,608 B |
| `public_vals` | 320 B |
| `proof` | 3,360 B |

> 与 README 声明（"~2 GB memory, ~8 s"）对齐：8 s 属实时。README 的 2 GB 是偏保守的估计，本机实测 241 MiB。

## 4. 关键标定数据（供 P4 外推）

这一节是本轮最有价值的产出——把"资源够不够"从猜测变成可算。

| 量 | 实测/标定值 | 来源 |
| --- | --- | --- |
| MNIST 电路行数 | 20,199 → 取 k=**15** | `estimate_size examples/mnist/config.msgpack` 输出 `Total number of rows: 20199`，与生成的 `15.params` 完全一致 |
| KZG 参数文件大小 | **≈ 2^k × 128 B** | k=15 → 4,194,564 B；2^15×128 = 4,194,304，差值为头部 |
| pkey 大小 | **≈ 2^k × 1120 B**（`num_cols=10` 配置下） | 36,725,924 / 32768 = 1120.5 |
| 证明耗时 | 8.27 s @ k=15 | — |
| 峰值 RSS | 241 MiB @ k=15 | — |

### 由此得到的资源外推公式（num_cols=10）

```
pkey_bytes  ≈ 1120 × 2^k
params_bytes ≈ 128 × 2^k
```

按此式，**pkey 大小对 k 呈指数增长**，这是本项目的硬约束（内存与磁盘双重）。

| k | 行数上限 | `params` 大小 | `pkey` 大小 | 80 GiB 内存是否可行 |
| --- | --- | --- | --- | --- |
| 15 | 32 K | 4.2 MB | 36.7 MB | ✅（已实测） |
| 20 | 1.0 M | 134 MB | 1.2 GB | ✅ |
| 22 | 4.2 M | 537 MB | 4.7 GB | ✅ |
| 24 | 16.8 M | 2.1 GB | 18.8 GB | ⚠️ 紧张（+MockProver 副本） |
| 26 | 67 M | 8.6 GB | 75 GB | ❌ 超内存 |
| 28 | 268 M | 34 GB | 300 GB | ❌ 远超 |

> 注：`time_circuit_kzg` 里除 pkey 外还有 `circuit.clone()` 多次、`MockProver::run` 的完整副本，实际内存约为 pkey 的 1.3–2 倍。因此 **k=24 是这台机器的现实上限，且留给后续步骤的空间很小**。

## 5. 已确认的上游代码风险（P3 会撞上）

读源码后发现三处对 ResNet18 不友好的点，均需在 P3 处理并单独提交：

1. **`MaxPool2D` 不接受 `SAME` padding**（`converter.py:158` 直接 `raise NotImplementedError`）。
   ResNet18 的 `MaxPool2d(3, stride=2, padding=1)` 经 TFLite 转换后就是 `SAME`，会直接中止转换。
   `Conv2D` 侧无此限制（Rust `out_hw` 支持 `PaddingEnum::Same`）。

2. **卷积只允许 `NONE/RELU/RELU6` 三种融合激活**（`valid_activations`）。
   TFLite 转换器默认会把 BN 折进卷积、把紧跟的 ReLU 融合；若某层 ReLU 没被融合，会变成独立的 `RELU` op——而转换器**没有** `RELU` 分支，会走 fallback 报错。

3. **`Add` 要求两个输入都已生成**（残差分支）；若其中一个被识别为常量张量，会走 `MaskNegInf` / 报错路径。

## 6. 结论

- P2 **完成**：构建成功（零补丁）、MNIST 的真实 KZG 证明与验证成功。
- 获得了可用的资源外推公式，P4 的可行性预检可以据此提前判断，不必先烧掉几小时。
- 下一步 P3 的真实风险不在 Rust 侧，而在 **TFLite 转换器对 ResNet18 图的接受度**（SAME padding 的 MaxPool 是确定的阻塞点）。
