Train:

```
CUDA_VISIBLE_DEVICES=1 python train_nlg.py --train_data train_queries_ddc.txt --val_data val_queries_ddc.txt --model_dir models > OUTPUT_FILE_1
```

Inference:

```
CUDA_VISIBLE_DEVICES=1 python inference_nlg.py --test_data test_queries_ddc.txt --model_dir models/T5_chat_ac_epoch_2.pth --inference_method beam > OUTPUT_FILE_2
```
choose inference_method between ```beam``` and ```top-p```
OUTPUT_FILE_2 saves the required input to eval script (to calculate metrics)

Evaluation:

```
python eval.py --output METRICS_FILE < OUTPUT_FILE_2
```
