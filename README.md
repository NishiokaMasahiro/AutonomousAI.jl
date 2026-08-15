# AutonomousAI.jl

**LLM を高次認知層として使い、Julia を計算・最適化・GPU 実行・システム統率の中心に置いた、
境界付き自律型 Computational Intelligence System。**

```
Goal → LLM(Plan) → World Model → Computational Optimizer → Julia Code/Kernel
     → Sandbox → Benchmark/Verify → CPU/CUDA → Hardware → Observation → Replan
```

---

## 0. まず最初に:このリポジトリの検証状態(重要)

本設計・実装は **Julia ランタイムが存在しない環境で作成された**。したがって:

| 項目 | 状態 |
|---|---|
| アーキテクチャ設計・数式定式化 | 完了 |
| Julia ソース(約 4,500 行) | 作成済み・**未コンパイル** |
| 構文チェック | 独自の block balance checker (`scripts/check_blocks.py`) のみ通過 |
| 単体テスト | 作成済み・**未実行** |
| ベンチマーク数値 | **一切測定していない。本リポジトリに実測値は 1 つも存在しない** |
| GPU パス | **未検証**(この環境に NVIDIA GPU なし) |

`docs/05_limitations_roadmap.md` に、何が検証済みで何が未検証かの完全な一覧がある。
性能表に架空の数値を埋めることは意図的に拒否した。**測っていない数字を書かないこと**は、
本システムが実装しようとしている原則そのものである(§34, §48)。

最初にやるべきことは:

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

### 1.1 応用例

本システムは「計算最適化器」だが、閉ループ制御・検証・サンドボックスを持つため、次のような応用に展開できる。

1. コーディングエージェント(自動実装・自動改善)
   与えられた目的(速度改善、メモリ削減、精度維持)に対して複数の実装候補を生成し、
   ベンチマークと検証を通過したものだけを採用する。重要なのは、
   「生成したコードをそのまま信じない」点で、性能と正しさの両方を計測で担保すること。

2. ソフトウェアセキュリティ(安全な自動変更パイプライン)
   AST allowlist 検証、能力モデル、資源上限、緊急停止を組み合わせることで、
   危険なコードパターンや過剰な権限行使を実行前に拒否できる。
   とくに、`eval`/`ccall`/外部実行などを含む不正な生成物を弾き、
   別プロセス実行で blast radius を限定する設計は、
   セキュアなコード生成・検証基盤として有効である。

3. 継続的最適化(CI/CD 連携)
   新しい入力分布やハードウェアに合わせて候補を再評価し、
   統計的に有意な改善のみを採用する。これにより、
   既存性能を壊さずに段階的な最適化を継続できる。

### 1.2 GitHub App / MCP 連携(HTTP + SSE)

GitHub Copilot Extensions の仕組みで本システムを配信する場合、
GitHub App から本リポジトリのブリッジサーバーにリクエストを転送し、
その背後で AI モデル推論を実行する構成を取れる。

- 実装ファイル: `scripts/copilot_extension_server.jl`
- 受信: ユーザープロンプト + コードコンテキスト(JSON)
- 出力: 通常 JSON (`/v1/infer`) または SSE ストリーム (`/v1/infer/stream`)

起動例:

```bash
julia --project=. scripts/copilot_extension_server.jl --host 127.0.0.1 --port 8081 --backend mock
```

主なエンドポイント:

1. `GET /health`
2. `POST /v1/infer`
3. `POST /v1/infer/stream`

`/v1/infer` リクエスト例:

```json
{
   "goal_id": "gh-ext-123",
   "prompt": "optimize matrix multiplication for this repository",
   "context": {
      "input_size": [4096, 4096],
      "repo": "owner/repo",
      "branch": "main"
   },
   "code_context": "src/Compute.jl: matmul candidate path"
}
```

`/v1/infer/stream` は SSE で `meta` → `delta` → `done` イベントを返す。

```bash
curl -N -X POST http://127.0.0.1:8081/v1/infer/stream \
   -H "content-type: application/json" \
   -d '{"prompt":"detect outliers and propose a safe plan","context":{"input_size":[2000000]}}'
```

バックエンド切替:

- `--backend mock` (既定, オフライン)
- `--backend openai` (OpenAI 互換 API)
- `--backend anthropic` (Anthropic Messages API)

環境変数:

- `AAI_LLM_BACKEND`, `AAI_SERVER_HOST`, `AAI_SERVER_PORT`, `AAI_SSE_CHUNK_MS`
- `AAI_OPENAI_URL`, `AAI_OPENAI_MODEL`, `OPENAI_API_KEY`
- `AAI_ANTHROPIC_MODEL`, `ANTHROPIC_API_KEY`

注意:

- 本サーバーは連携用の最小実装であり、認証・認可・監査ログ保存・レート制限は別途必要。
- GitHub App 本番運用時は、App 側で署名検証とトークン検証を必ず行うこと。
- MCP サーバーとして提供する場合も、同じ推論ロジックをツール実装に再利用できる。

---

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

**仕様からの変更(§4 は変更を許可、理由の明示を要求している):**

1. `Core` → `AgentCore`。ネストした `Core` は Julia の `Core` を隠蔽し `using .Core` が
   曖昧になる。
2. `Reasoning` は `LLM.jl` + `AgentCore.jl` に分割。Planner を LLM 抽象から分離しないと
   provider 交換時に planner まで書き換えになる。
3. `Execute.jl` を追加。「生成コードを測る駆動コード」は生成物でも計測器でもない第三の
   信頼カテゴリであり、混ぜると計測結果の信頼性が崩れる。
4. `Sandbox.jl` を `Safety` から独立。隔離は OS の仕事でありポリシー判断とは層が違う。

---

## 4. この設計が仕様に対して主張する 5 つの変更

詳細は `docs/06_design_review.md`。要約:

| # | 仕様の記述 | 本実装 | 理由 |
|---|---|---|---|
| D2 | 「生成コードを sandbox で実行」 | **同一プロセス内 sandbox は不可能**。別 OS プロセス + ulimit + (任意で)コンテナ | Julia は自分自身を sandbox 化できない(`eval`/`ccall`/マクロ展開) |
| D6 | `Cost = αT + βM + γE + δErr + εRisk` | 正規化 + **hard constraint は重みでなく実行可能領域**として分離 | 秒とバイトの加算は次元不整合。重みは十分な利得で常に突破される |
| D8 | 「speedup < 2% で停止」 | **bootstrap 信頼区間の下限**が閾値を超えるかで判定 | 実機のノイズは日常的に 2% を超える |
| D9 | `rel_err = |r-ref|/|ref|` | `|r-ref| ≤ atol + rtol|ref|`、許容誤差は精度と n から導出 | `ref≈0` で未定義。FP32 で 1e-10 を要求するのは無意味 |
| D7 | GPU を FP64 の基準に | **CPU を FP64 oracle に**。GeForce の FP64 は FP32 の 1/64 | RTX 5070 で FP64 参照を取るのは最も遅い経路 |

---

## 5. ドキュメント

| ファイル | 内容 |
|---|---|
| `docs/01_gpt3_analysis.md` | Brown et al. (2020) の限界分析と、各限界への構造的対応 |
| `docs/02_architecture.md` | 層構造・データフロー・型設計・Julia 機能の使い所 |
| `docs/03_mathematics.md` | エージェント状態モデル、コストモデル、roofline、UCB、GP-EI、停止則 |
| `docs/04_safety.md` | 脅威モデル、TCB、能力モデル、制約階層、自己改変ポリシー |
| `docs/05_limitations_roadmap.md` | 既知の限界、失敗モード分析、未検証事項、ロードマップ |
| `docs/06_design_review.md` | **仕様に対する批判的レビュー(D1–D14)** |
| `docs/07_gpt3_comparison.md` | GPT-3 との能力比較表(誇張のない版) |

---

## 6. ライセンス

本リポジトリは **MIT License** で提供する。
詳細は `LICENSE` を参照。


