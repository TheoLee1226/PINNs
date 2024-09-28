torch 2.2.0
cuda 1.2.1
python 3.9.13

torch的lib建議手動安裝，放在TorchLib資料夾 

1. DampedOscillation_Data 生成阻尼震盪原始資料，模擬實驗數據。執行會生成 Data.csv 供應主程式使用。
2. DampedOscillation_PINNs 第一版，邊界條件僅有 x_0
3. DampedOscillation_PINNs_v2 第二版，邊界條件有 x_0 v_0，加上正確率圖表