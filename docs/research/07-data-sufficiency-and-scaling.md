# 07. longf のデータ充足性・スケーリング調査

期間: 2026-08-25 / 対象: GitHub Issue #7

## 事前固定プロトコル

結果を見る前に、一次実験を次のとおり固定した。

- 基準方策: `longf` PPO + MLP。9ペア日足、同一特徴量、`n_steps=1024`、
  `batch_size=1024`、300,000 requested decisions、3シード(42/43/44)
- 評価: 既存walk-forwardの2019〜2025年7フォールド。各フォールドで直前6ヶ月を
  validation、評価年をholdoutとし、評価結果でcheckpoint・条件・seedを選ばない
- 操作変数: train終端を固定し、train開始だけを評価年ごとの2年前・4年前・8年前・
  expanding開始に変更する
- 実行: 全84 runを同じRTX A6000上のCUDAで再学習する。過去のCPU runは混ぜない
- 一次指標: 同一fold・seedで対にした評価年全体のnet cumulative log return
- 二次指標: gross return、annualized Sharpe、max drawdown、cost、gross leverage、
  3-seed action-mean ensemble、およびfold内seed標準偏差
- 独立情報量: raw timestamp、feature warmup後timestamp、到達可能な固有market
  transition、128-step episode開始候補、実行env decisions、実測完了episode、再利用率を
  別々に報告する。9ペア同時観測を独立な9標本とは数えない
- ESS: pair別のdaily log return、absolute/squared return、mom24、xr_mom24に対し、
  最大lag 252のGeyer initial-positive-pair推定を使う。負の相関で `ESS>N` となる場合は
  `N` を上限とする。横断依存はreturn相関行列のparticipation ratioを別途示す
- 不確実性: 7個の評価年を再標本化するfold bootstrapと、隣接2年を単位とするmoving-block
  bootstrapを併記する。日次barをIIDとして信頼区間を作らない

PPOの `total_timesteps` は下限である。8 env × 1,024 steps単位のため、各runは37 rollout、
303,104 env decisions、既定10 epochで2,960 optimizer minibatch updates、3,031,040 sample
presentationsとなる。128-step episodeが完走する場合の換算は2,368本だが、最終表では
マージンコールを含む実測完了数を使う。

### データ品質とlineage

元のgitignored cacheが環境になかったため、`configs/fetch_1d9p_2003.yaml` からyfinanceを
再取得し、FRED金利を60日lagで付与した。再取得値には桁違いのOHLCと翌日全戻し型の
不良printが残っていたため、`forex-clean-spikes` の一般ルールで修復する。修復件数を14行に
固定し、providerの改訂で件数が変わった場合は実験を停止する。実験成果物にはraw/clean/carry
cacheのSHA-256を記録する。

## スコープ境界

ペア数を変えると現MLPの入力次元・action次元・parameter数・`xz/xr`特徴量の意味まで変わり、
cross-sectional breadthと方策容量を分離できない。これはarchitecture searchを禁止する
IssueのNon-goalと衝突するため、一次因果実験には含めず、既存の7/9/10ペア結果を探索的証拠
としてのみ扱う。

同様に既存時間足実験はユニバース、特徴量、観測窓、decision intervalが異なるため、日足との
controlled comparisonではない。高頻度化に関する結論は既存結果の限定的な再解釈とし、
新しいfrequency experimentは別Issueの対象とする。

## 結果

実験完了後に機械生成された表・曲線と結論を追記する。
