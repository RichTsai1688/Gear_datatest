# 齒輪特徵分類專案指南

本文整合本次特徵工程、模型微調與報表/檢視器更新的全部步驟，提供自資料準備到結果檢視的一條龍流程。主要成果：以 39 維延伸特徵驅動的 MLP，在測試集達成 **98.4% 準確率、97.9% 精確率、100% 召回率**（閾值 0.16）。

---

## 1. 環境需求
- Python ≥ 3.9（TensorFlow、NumPy 等已於環境安裝）
- Node.js 18+（用於 Viewer）
- 專案根目錄：`/data2/codespace/gear_classificaiton`

---

## 2. 原始資料與資料集
- 原始振動檔：`gear_raw_data/`（CSV、含時間與 X/Y/Z 軸）
- 預先打包：`datasets/cw_dataset.npz`（可由 `build_datasets.py` 或提供的 ZIP 取得）
- 類別：`good` = 0、`bad` = 1，共 822 筆樣本（192 good / 630 bad）
- 切分檔：`datasets/feature_splits.json`
  - Train 575（good 134 / bad 441）
  - Validation 122（good 28 / bad 94）
  - Test 125（good 30 / bad 95）

> **注意**：`feature_splits.json` 內含 row_index、sample_index、label，用以確保每次重跑流程皆沿用相同範例。

---

## 3. 特徵萃取
腳本：`build_feature_dataset.py`

### 3.1 特徵類型
| 類別 | 指標 |
|------|------|
| 時域（各軸） | mean、std、RMS、crest factor、peak-to-peak、skewness、kurtosis、energy |
| 頻域（各軸） | peak frequency、peak magnitude (dB)、spectral centroid、bandwidth |
| 軸間相關 | XY、XZ、YZ 皮爾森相關係數 |

### 3.2 特徵組合
- **Top 10**：手動挑選的 10 個強判別特徵（原始模型使用）
- **Extended 39**：完整 39 維特徵（本次最佳模型使用）

### 3.3 執行指令
```bash
# 產生 Top 10 特徵（預設）
python build_feature_dataset.py datasets/cw_dataset.npz \
  --output datasets/cw_top_features.csv \
  --summary-json reports/cw_feature_summary.json

# 產生 Extended 特徵
python build_feature_dataset.py datasets/cw_dataset.npz \
  --feature-set extended \
  --output datasets/cw_extended_features.csv \
  --summary-json reports/cw_extended_feature_summary.json
```

輸出：
- CSV（含 sample_index、source、label、特徵）
- JSON（每個特徵的 good/bad 均值、標準差與差異）

---

## 4. 特徵比較重點（Extended set）
- X 軸 crest factor +8.07、kurtosis +794 → 壞品震動尖峰明顯增加
- Z 軸 peak frequency +35.2 Hz、bandwidth +100.5 Hz → 壞品高頻能量擴散
- XY 相關係數降低（-0.0282）→ 軸向耦合關係改變
- Y 軸 spectral centroid -107.7 Hz → 能量偏向低頻
詳見 `reports/cw_extended_feature_summary.json` 或 `feature-method.md`

---

## 5. 模型訓練與調參
腳本：`train_feature_network.py`

### 5.1 重要參數
- `--dataset`：特徵 CSV（預設 `datasets/cw_top_features.csv`）
- `--hidden-sizes`：MLP 隱藏層，例如 `512 256 128`
- `--auto-class-weight`：啟動類別權重（逆頻率）
- `--threshold-metric`：驗證集閾值最佳化依據（accuracy/balanced_accuracy/f1）

### 5.2 最佳訓練指令
```bash
python train_feature_network.py \
  --dataset datasets/cw_extended_features.csv \
  --epochs 200 --batch-size 32 \
  --dropout 0.2 --hidden-sizes 512 256 128 \
  --learning-rate 3e-4 --patience 30 --plateau-patience 15 \
  --auto-class-weight --threshold-metric accuracy
```

### 5.3 模型細節
- 架構：Dense(512) → Dense(256) → Dense(128) → Sigmoid，ReLU + BatchNorm + Dropout(0.2)
- L2 regularization：1e-4
- 類別權重：good 2.145、bad 0.652
- Checkpoint：`checkpoints/feature_mlp_best.weights.h5`
- 訓練報告：`reports/feature_mlp_report.json`

### 5.4 閾值調校
- 驗證最佳閾值：**0.16**（以 accuracy 99.18% 最佳化）
- 混淆矩陣：
  - Train：TN=134, TP=441, FP=0, FN=0 → 100%
  - Val：TN=28, TP=93, FP=0, FN=1 → 99.18%
  - Test：TN=28, TP=95, FP=2, FN=0 → 98.4%
  - Overall：TN=190, TP=629, FP=2, FN=1 → 99.64%

> 閾值及詳細指標可於 `reports/feature_mlp_report.json` 與 Viewer 中查詢。

---

## 6. Viewer 更新
- 位置：`viewer/`
- 新增 API `/api/model/summary`：
  - 回傳特徵 CSV 欄位、Top 10 列表、總體/切分樣本數、類別權重、驗證與測試指標、閾值等
- 前端頁面於上方增加 **Model Summary** 卡片，展現上述資訊
- Viewer 啟動方式：
  ```bash
  cd viewer
  npm install  # 首次
  npm start    # 或 node server.js
  ```
- 預設埠：`http://localhost:3000`

---

## 7. 文件總覽
- `finalize-report.md`：本次專案成果總結與建議
- `feature-method.md`：特徵萃取方法與 good/bad 比較說明
- `README.md`（本文）：全流程操作指南

---

## 8. 延伸建議
1. 於其它方向/機況資料重複流程，驗證 0.16 閾值是否適用
2. 若未來擴充特徵（如小波、包絡分析），請同步更新 `build_feature_dataset.py` 與本文檔案
3. 併入 FFT 模型（CNN/LSTM）作為 ensemble，可提升壞品偵測可靠度

---

祝使用順利！若需自動化流程，建議將上述指令包裝為 Makefile 或 CI-Workflow，確保重新訓練時步驟一致。
