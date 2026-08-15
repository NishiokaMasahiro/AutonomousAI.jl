# AutonomousAI.jl

**LLM を高次認知層として使い、Julia を計算・最適化・GPU 実行・システム統率の中心に置いた、
境界付き自律型 Computational Intelligence System。**

```
Goal → LLM(Plan) → World Model → Computational Optimizer → Julia Code/Kernel
     → Sandbox → Benchmark/Verify → CPU/CUDA → Hardware → Observation → Replan
```

---

## 0. まず最初に

最初にやること:

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'   # 依存は stdlib のみ
julia --project=. test/runtests.jl                     # ここで初めて実測が始まる
```

---

## 1. クイックスタート

```julia
using AutonomousAI

# ハードウェア観測(読み取り専用)
AutonomousAI.Interface.main(["hardware"])

# 閉ループ実行
agent  = AutonomousAI.AgentCore.Agent()
goal   = AutonomousAI.AgentCore.Goal("センサ列の異常値を可能な限り高速に検出せよ"; input_size = [2_000_000], n_calls = 200)
report = AutonomousAI.AgentCore.run_goal!(agent, goal)
AutonomousAI.Interface.print_report(report)
```

CLI:

```bash
julia --project=. -e 'using AutonomousAI; AutonomousAI.Interface.main(ARGS)' -- hardware
julia --project=. -e 'using AutonomousAI; AutonomousAI.Interface.main(ARGS)' -- demo "detect outliers"
julia --project=. examples/self_improvement.jl      # §46 の 5 段階比較
julia --project=. examples/out_of_core.jl           # §39 の 100 GB ワークフロー
julia --project=. scripts/copilot_extension_server.jl --port 8081  # Copilot Extensions / GitHub App 連携用
```

## 2. 依存関係の方針

コアは **stdlib のみ**。CUDA.jl / KernelAbstractions.jl / BenchmarkTools.jl は
package extension による **weak dependency** であり、無い場合は
「候補空間からその能力が消える」だけでシステムは動く。

理由: エージェントが監査可能であること、レジストリのないマシンで起動できることは
安全性要件である。JSON パーサすら自前(`MiniJSON`)なのはこのため。

Python は使用していない。使用する場合の条件は `docs/06_design_review.md` の D14 を参照。

---

## 3. モジュール構成(仕様 §4 からの変更点は理由付き)

```
src/
  MiniJSON.jl        依存ゼロ JSON(LLM 境界)
  Schema.jl          閉じた行動空間 + 検証           ← 仕様 §38 の中核
  HAL.jl             CPU/GPU/Memory/Storage 抽象化   ← 仕様 §27
  WorldModel.jl      SystemState / HardwareState     ← 仕様 §22
  Memory.jl          5 種の記憶(分離)              ← 仕様 §2.2
  Safety.jl          PolicyEngine / Limits / E-Stop  ← 仕様 §29
  CodeGeneration.jl  テンプレート合成 + AST 変換 + 静的検証 ← 仕様 §12,14,17,21
  Compute.jl         backend 抽象 + 信頼できる参照実装 ← 仕様 §13
  Sandbox.jl         別プロセス実行                  ← 仕様 §30
  Verification.jl    数値検証・性質テスト            ← 仕様 §34
  Optimization.jl    cost model / roofline / UCB / GP / 停止則 ← 仕様 §9,10,11,15,33
  Execute.jl         sandbox driver + 検証ブリッジ
  LLM.jl             LLM provider 抽象              ← 仕様 §36
  AgentCore.jl       Goal / Decision / 閉ループ      ← 仕様 §24,25
  Interface.jl       CLI / レポート
```

**仕様からの変更:**

1. `Core` → `AgentCore`。ネストした `Core` は Julia の `Core` を隠蔽し `using .Core` が
   曖昧になる。
2. `Reasoning` は `LLM.jl` + `AgentCore.jl` に分割。Planner を LLM 抽象から分離しないと
   provider 交換時に planner まで書き換えになる。
3. `Execute.jl` を追加。「生成コードを測る駆動コード」は生成物でも計測器でもない第三の
   信頼カテゴリであり、混ぜると計測結果の信頼性が崩れる。
4. `Sandbox.jl` を `Safety` から独立。隔離は OS の仕事でありポリシー判断とは層が違う。

---

## 4. ライセンス

本リポジトリは **MIT License** で提供する。
詳細は `LICENSE` を参照。


