# 數學化說明與流程重點

## 1. 模型輸出與決策閾值
### 1.1 Sigmoid 輸出
模型最後一層為 Sigmoid 神經元，對於輸入向量 \(\mathbf{x}\) 產生介於 \([0,1]\) 之間的實數 \(p\)：
\[
p = \sigma(z) = \frac{1}{1 + e^{-z}}
\]
其中 \(z\) 為線性組合 \(z = \mathbf{w}^T \mathbf{x} + b\)。在分類觀點可視為「樣本屬於壞品 (label=1) 的機率估計」。

### 1.2 閾值轉換
為取得二元結果 \(\hat{y} \in \{0,1\}\)，需選定決策閾值 \(\tau\)：
\[
\hat{y} =
\begin{cases}
1, & p \ge \tau \\
0, & p < \tau
\end{cases}
\]

### 1.3 閾值搜尋
程式 (`find_best_threshold`) 在區間 \([0.05, 0.95]\) 以固定步長掃描候選閾值 \(\{\tau_i\}\)，對每個 \(\tau_i\) 重新產生預測 \(\hat{y}(\tau_i)\)，並計算指定指標（本研究採用 validation accuracy）：
\[
\text{Accuracy}(\tau_i) = \frac{\text{TP}(\tau_i) + \text{TN}(\tau_i)}{\text{TP}(\tau_i) + \text{TN}(\tau_i) + \text{FP}(\tau_i) + \text{FN}(\tau_i)}
\]

其中：
- TP (True Positive)：真正例數量，亦即壞品且被判為壞品；
- TN (True Negative)：真負例；
- FP (False Positive)：偽正例；
- FN (False Negative)：偽負例。

取使評估指標最大的閾值 \(\tau^*\)。於驗證集中獲得 \(\tau^* = 0.16\)，是後續報告的參考值。

## 2. 資料與特徵流程
1. **原始數據**：三軸振動 CSV (`gear_raw_data/`)，透過 `build_datasets.py` 或既有封裝產生 `datasets/cw_dataset.npz`。
2. **切割**：以 stratified split 建立訓練/驗證/測試分配（`datasets/feature_splits.json`），保留索引與標籤資訊。
3. **特徵萃取** (`build_feature_dataset.py`)：
   - 時域指標：mean、std、RMS、crest factor、peak-to-peak、skewness、kurtosis、energy；
   - 頻域指標：dominant frequency、peak magnitude (dB)、spectral centroid、bandwidth；
   - 軸間相關：XY、XZ、YZ。
   - 產出「Top10」與「Extended 39」兩種特徵 CSV 與對應 summary JSON。
4. **標準化** (`train_feature_network.py`)：對訓練集計算均值與標準差，並套用到驗證與測試資料。

## 3. 模型訓練流程
1. **架構**：Dense 層 512→256→128→Sigmoid；中間層搭配 ReLU、BatchNorm、Dropout(0.2)。
2. **損失與優化**：Binary Cross-Entropy、Adam (lr=3e-4)，採 ReduceLROnPlateau 與 EarlyStopping 控制學習率與過擬合。
3. **類別權重**：
   \[
   w_{\text{class}} = \frac{N}{K \cdot N_{\text{class}}}
   \]
   其中 \(N\) 為總樣本數、\(K\) 為類別數（2）、\(N_{\text{class}}\) 為該類別樣本量。計算結果 good=2.145、bad=0.652。
4. **指令**：
   ```bash
   python train_feature_network.py \
     --dataset datasets/cw_extended_features.csv \
     --epochs 200 --batch-size 32 \
     --dropout 0.2 --hidden-sizes 512 256 128 \
     --learning-rate 3e-4 --patience 30 --plateau-patience 15 \
     --auto-class-weight --threshold-metric accuracy
   ```
5. **成果輸出**：
   - 最佳權重：`checkpoints/feature_mlp_best.weights.h5`
   - 報告：`reports/feature_mlp_report.json`（含歷程、閾值調校）
   - Viewer summary：透過 `/api/model/summary` 供前端顯示

## 4. 混淆矩陣與指標
以最佳閾值 \(\tau^* = 0.16\) 評估：

| Split      | TN | TP | FP | FN | Accuracy | Precision | Recall | Specificity | F1 |
|------------|----|----|----|----|----------|-----------|--------|-------------|----|
| Train      |134 |441 |0   |0   |1.000     |1.000      |1.000   |1.000        |1.000 |
| Validation |28  |93  |0   |1   |0.992     |1.000      |0.989   |1.000        |0.995 |
| Test       |28  |95  |2   |0   |0.984     |0.979      |1.000   |0.933        |0.990 |

### 指標定義
\[
\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}} \quad ; \quad
\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}} \quad ; \quad
\text{Specificity} = \frac{\text{TN}}{\text{TN} + \text{FP}}
\]
\[
\text{F1} = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}
\]

## 5. 系統整合與視覺化
1. Viewer 伺服器 (`viewer/server.js`) 新增 `/api/model/summary`，整合特徵 CSV、分割資訊及 `feature_mlp_report.json`。
2. 前端 (`viewer/public/index.html`、`main.js`、`styles.css`) 新增 Model Summary 卡片，顯示特徵欄位、Top10 列表、閾值與指標表格。
3. 執行方式：
   ```bash
   cd viewer
   npm install
   npm start  # http://localhost:3000
   ```

## 6. 總結
- 擴充至 39 維特徵，保留振動訊號的高頻變化與軸向耦合資訊。
- 類別權重搭配閾值調校，於避免漏報（FN=0）的前提下維持高準確率。
- 數據與模型細節均已文檔化（`README.md`、`feature-method.md`、`finalize-report.md`、本文件），利於重現與後續拓展。

