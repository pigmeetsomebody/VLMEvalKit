## 贪心搜索视觉token动态化跳层

## 环境准备

除了按照VLMEvalKit官方教程按照所需依赖外，因为我们的视觉标记跳层方法对transformers中modeling_qwen2_5_vl.py文件进行了修改，因此，需要安装改动后的transformers库, 运行如下命令:

```
pip install -e ./third_party/transformers

```


## 贪心搜索脚本（带scalar）

```

python qwen_beam_search_layer.py --checkpoint /data/share/Qwen2.5-VL-32B-Instruct/ --dataset textvqa_val --n-samples 20 --beam_width 1 --target_saving_benefits 20

```

- checkpoint: 模型目录
- dataset: 校准数据集
- beam_width: 束搜索参数(=1时为贪心搜索)
- target_saving_benefits: 需要剪枝的层数（MLP为1.0， Attention: 0.5）

返回输出:
```
=== FINAL POLICY ===
Final Score: 0.007415771484375
Skipped MLPs: [27, 5, 16, 23, 21, 24, 25, 19, 18]
Skipped Attns: []
Scalars: {27: 0.9066757559776306, 5: 1.0371835231781006, 16: 0.988783597946167, 23: 1.0793545246124268, 21: 1.0308176279067993, 24: 1.1073952913284302, 25: 1.1257060766220093, 19: 0.9927554130554199, 18: 1.0417094230651855}
```

得到贪心搜索剪枝脚本后，修改模型config(/data/share/Qwen2.5-VL-7B-Instruct/config.json)

```
{
  "architectures": [
    "Qwen2_5_VLForConditionalGeneration"
  ],
  "attention_dropout": 0.0,
  "bos_token_id": 151643,
  "eos_token_id": 151645,
  "vision_start_token_id": 151652,
  "vision_end_token_id": 151653,
  "vision_token_id": 151654,
  "image_token_id": 151655,
  "video_token_id": 151656,
  "hidden_act": "silu",
  "hidden_size": 3584,
  "initializer_range": 0.02,
  "intermediate_size": 18944,
  "max_position_embeddings": 128000,
  "max_window_layers": 28,
  "model_type": "qwen2_5_vl",
  "num_attention_heads": 28,
  "num_hidden_layers": 28,
  "num_key_value_heads": 4,
  "rms_norm_eps": 1e-06,
  "rope_theta": 1000000.0,
  "sliding_window": 32768,
  "tie_word_embeddings": false,
  "torch_dtype": "bfloat16",
  "transformers_version": "4.41.2",
  "use_cache": true,
  "use_sliding_window": false,
  "vision_config": {
    "depth": 32,
    "hidden_act": "silu",
    "hidden_size": 1280,
    "intermediate_size": 3420,
    "num_heads": 16,
    "in_chans": 3,
    "out_hidden_size": 3584,
    "patch_size": 14,
    "spatial_merge_size": 2,
    "spatial_patch_size": 14,
    "window_size": 112,
    "fullatt_block_indexes": [
      7,
      15,
      23,
      31
    ],
    "tokens_per_second": 2,
    "temporal_patch_size": 2
  },
  "rope_scaling": {
    "type": "mrope",
    "mrope_section": [
      16,
      24,
      24
    ]
  },
  "vocab_size": 152064,
  "skip_layer_mlp": [27, 5, 16, 23, 21, 24, 25, 19, 18],
  "skip_layer_attn": [],
  "mlp_scalars": {
    "27": 0.9066757559776306, "5": 1.0371835231781006, "16": 0.988783597946167, "23": 1.0793545246124268, "21": 1.0308176279067993, "24": 1.1073952913284302, "25": 1.1257060766220093, "19": 0.9927554130554199, "18": 1.0417094230651855}
}

```

## 运行评测脚本

```
CUDA_VISIBLE_DEVICES=0 torchrun --rdzv_endpoint=localhost:29413  run.py --data MME MMBench_DEV_EN MMBench_DEV_CN ChartQA_TEST TextVQA_VAL MMMU_DEV_VAL OCRBench  --model Qwen2.5-VL-7B-Instruct --verbose --work-dir output_16_scalars

```