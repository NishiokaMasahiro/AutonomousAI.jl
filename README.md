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
julia --project=. -e 'using Pkg; Pkg.instantiate()'   # 依存をインストール
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
julia --project=. benchmark/llm_model_comparison.jl # Mimase / Opus 5 / Gemini 3.7 比較
```

## 1.1 LLM モデル名: Mimase

AutonomousAI の既定 LLM は **Mimase** とする。

- 実装上は `AutonomousAI.LLM.MimaseLLM()` を既定として使用
- 旧名 `MockLLM` は後方互換エイリアスとして残す
- 役割は「実行コード生成」ではなく「Schema に沿った計画提案」に限定

## 1.2 Mimase と Opus 5 / Gemini 3.7 の比較方法

比較対象:

1. Mimase (`MimaseLLM`)
2. Anthropic Opus 5 (`AnthropicLLM`)
3. Gemini 3.7 (`OpenAICompatibleLLM` 経由)

実行:

```bash
set ANTHROPIC_API_KEY=...
set GEMINI_API_KEY=...
set AAI_OPUS_MODEL=claude-opus-5
set AAI_GEMINI_MODEL=gemini-3.7-pro
set AAI_GEMINI_URL=https://generativelanguage.googleapis.com/v1beta/openai/chat/completions
julia --project=. benchmark/llm_model_comparison.jl
```

評価指標:

- transport success rate (API 到達率)
- schema success rate (Schema.parse_plan 通過率)
- latency median / p95
- plan quality score (benchmark/verify/execute の網羅と順序)

結果出力:

- コンソールに集計表を表示
- `benchmark/llm_model_comparison_results.csv` に詳細行を保存

## 2. 依存関係と実行要件（重要）

- 本プロジェクトの並列計算は **JACC.jl 必須**（フォールバックなし）。
- GPU 実行は **CUDA.jl 必須**。
- `Compute` 層では `JACC.@init_backend` によりバックエンド初期化を行う。

### セットアップ

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. -e 'using Pkg; Pkg.precompile()'
```

### CUDA 環境要件

- NVIDIA Driver が CUDA ランタイム要件を満たしていること。
- CUDA ランタイム不一致（例: 13.1 / 13.2）が出る場合は、ランタイムを固定して Julia を再起動:

```julia
using CUDA
CUDA.set_runtime_version!(v"13.1")  # 環境に合わせて変更
```

その後、Julia を再起動して再 precompile:

```bash
julia --project=. -e 'using Pkg; Pkg.precompile()'
julia --project=. test/runtests.jl
```

---

## 3. モジュール構成

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

---

## 4. ライセンス

本リポジトリは **MIT License** で提供する。
詳細は `LICENSE` を参照。


