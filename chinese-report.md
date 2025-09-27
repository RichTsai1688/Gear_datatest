# 齒輪特徵分類之神經網路微調研究報告

## 摘要
本研究針對齒輪製程所蒐集之三軸振動訊號進行特徵工程與神經網路微調，旨在提升壞品偵測的準確性。透過擴充手動分析所得的 Top 10 特徵為 39 維延伸特徵集，並導入類別權重與驗證集閾值調校，最終多層感知器模型於測試集達成 98.4% 準確率、97.9% 精確率與 100% 召回率，驗證閾值為 0.16。本報告整合資料處理、特徵萃取、模型訓練與視覺化檢視流程，作為後續部署與研究延伸之參考。

## 一、緒論
齒輪製造過程中之良品與壞品具有微幅但關鍵的振動差異。為降低現場檢測成本與人為誤判，本研究建立以振動訊號為基礎的自動分類流程。初期模型僅採用 10 項手動挑選的特徵，表現受限。本次工作聚焦於兩大改進：
1. 擴充特徵維度，保留更多振動動態；
2. 透過類別權重與驗證閾值調校，抑制資料不平衡帶來的偏差。

## 二、資料集與前處理
- 原始資料：`gear_raw_data/`，每筆樣本包含時間與 X/Y/Z 軸振動紀錄。
- 預處理檔：`datasets/cw_dataset.npz`（clockwise 方向），內含 `signals`、`lengths`、`labels`、`sources`、`label_names`。
- 樣本數：共 822 筆，其中良品 (good) 192 筆、壞品 (bad) 630 筆。
- Stratified split（`datasets/feature_splits.json`）：
  - 訓練集 575（good 134 / bad 441）
  - 驗證集 122（good 28 / bad 94）
  - 測試集 125（good 30 / bad 95）
該分割檔記錄 row_index 與 source，可完全重現每次實驗。

## 三、特徵萃取方法
### 3.1 特徵家族
- **時域指標**：各軸 mean、std、RMS、crest factor、peak-to-peak、skewness、kurtosis、energy。
- **頻域指標**：各軸 dominant frequency、peak magnitude (dB)、spectral centroid、bandwidth。
- **軸間相關**：XY、XZ、YZ 相關係數。

### 3.2 特徵組合
- **Top 10**（原始模型）：針對 X/Y/Z 軸挑選的 10 項判別力強之指標。
- **Extended 39 維**：整合上述所有時域、頻域與相關特徵，總計 39 欄。
- `build_feature_dataset.py` 在執行時可透過 `--feature-set` 切換輸出，並同步產生每個特徵在良品與壞品的均值、標準差與差異資料。

### 3.3 特徵差異觀察
依 `reports/cw_extended_feature_summary.json` 可歸納以下重點：
- 壞品在 X 軸之 crest factor (+8.07) 與 kurtosis (+794.50) 顯著增大，顯示尖峰震動加劇；
- Z 軸 peak frequency (+35.2 Hz) 與 bandwidth (+100.5 Hz) 提升，代表高頻能量擴散；
- Y 軸 spectral centroid 減少 107.7 Hz，能量向低頻移動；
- XY 相關係數下降（−0.0282），反映軸間耦合退化；
- Skewness 值改變，壞品震動分布的非對稱性明顯不同。

## 四、模型架構與訓練策略
### 4.1 模型設定
- 網路結構：Dense(512) → Dense(256) → Dense(128) → Sigmoid，層間搭配 ReLU、Batch Normalization 與 Dropout(0.2)。
- 正規化：L2=1e-4；ReduceLROnPlateau (factor=0.5, patience=15)；EarlyStopping (patience=30)。
- 類別權重：good 2.145、bad 0.652，解決資料不平衡問題。
- 標準化：以訓練集均值/標準差進行 z-score，並套用至驗證與測試集。

### 4.2 訓練指令
```bash
python train_feature_network.py \
  --dataset datasets/cw_extended_features.csv \
  --epochs 200 --batch-size 32 \
  --dropout 0.2 --hidden-sizes 512 256 128 \
  --learning-rate 3e-4 --patience 30 --plateau-patience 15 \
  --auto-class-weight --threshold-metric accuracy
```
訓練後將最佳權重儲存於 `checkpoints/feature_mlp_best.weights.h5`，並生成報告 `reports/feature_mlp_report.json`。

### 4.3 閾值調校
透過驗證集搜尋閾值 0.05–0.95，選擇使 accuracy 最大者。最佳閾值為 **0.16**，對應的驗證測試結果同時寫入報告與 Viewer summary。

## 五、實驗結果
| Split      | TN | TP | FP | FN | Accuracy | Precision | Recall | Specificity | F1   |
|------------|----|----|----|----|----------|-----------|--------|-------------|------|
| Train      |134 |441 |0   |0   |1.000     |1.000      |1.000   |1.000        |1.000 |
| Validation |28  |93  |0   |1   |0.992     |1.000      |0.989   |1.000        |0.995 |
| Test       |28  |95  |2   |0   |0.984     |0.979      |1.000   |0.933        |0.990 |
| Overall    |190 |629 |2   |1   |0.996     |0.997      |0.998   |0.990        |0.998 |

模型在測試集保持高召回率（0.16 閾值下無 FN），僅產生 2 筆誤報，滿足壞品全面攔截的現場需求。

## 六、視覺化與系統更新
- Viewer (`viewer/`) 新增 `/api/model/summary`，可讀取 extended 特徵 CSV、分割資料、類別權重與閾值調校結果。
- 前端介面新增 Model Summary 卡片，呈現特徵欄位、Top 10 清單、策略指標與混淆矩陣。
- 啟動方式：`cd viewer && npm install && npm start`，預設位於 `http://localhost:3000`。

## 七、結論與未來工作
本研究證明，結合延伸特徵集、類別權重與閾值調校之 MLP，可於齒輪振動資料中達到高準確率與零漏報的壞品偵測效果。未來可進一步：
1. 在不同運轉方向或新批次資料重複流程，驗證閾值 0.16 的泛化能力；
2. 與 CNN/LSTM 等頻譜模型進行 Ensemble，以提升對未知異常的魯棒性；
3. 納入額外特徵（如小波或包絡分析），並同步更新 `build_feature_dataset.py` 與本文檔案。

透過本報告與附帶的 `README.md`、`feature-method.md`、`finalize-report.md`，即能完整重現資料處理、模型訓練與檢視流程。

