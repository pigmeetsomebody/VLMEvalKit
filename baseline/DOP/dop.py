import torch
import torch.nn as nn
from transformers.models.qwen2.modeling_qwen2 import Qwen2MLP 
import os
import sys

# --- 上下文管理器 ---
class DopContext:
    _visual_mask = None
    
    @classmethod
    def set_mask(cls, mask):
        cls._visual_mask = mask
        
    @classmethod
    def get_mask(cls):
        return cls._visual_mask

# --- 核心组件1: 支持剪枝的 MLP ---
class DOPQwenMLP(nn.Module):
    def __init__(self, original_mlp, layer_idx, pruning_depth_dp):
        super().__init__()
        # 标准 Qwen2 结构
        self.gate_proj = original_mlp.gate_proj
        self.up_proj = original_mlp.up_proj
        self.down_proj = original_mlp.down_proj
        self.act_fn = original_mlp.act_fn
        
        self.layer_idx = layer_idx
        self.dp = pruning_depth_dp 

    def forward(self, x):
        # 1. 标准计算
        gate_out = self.gate_proj(x)
        up_out = self.up_proj(x)
        intermediate = self.act_fn(gate_out) * up_out
        output = self.down_proj(intermediate)
        
        # 2. 剪枝逻辑 
        # 当当前层数超过设定的保留深度(dp)时，对视觉Token进行mask
        if self.layer_idx >= self.dp:
            mask = DopContext.get_mask()
            if mask is not None:
                # 检查序列长度，只在 Prefill 阶段应用
                curr_len = output.shape[1]
                mask_len = mask.shape[1]
                
                if curr_len == mask_len:
                    mask = mask.to(output.device).to(output.dtype)
                    mask_broadcast = mask.unsqueeze(-1)
                    # 将视觉部分的输出置零 (模拟剪枝)
                    output = output * (1.0 - mask_broadcast)
        
        return output

# --- 核心组件2: 模型注入工具 ---
def apply_dop_patch(model, target_dp):
    """
    target_dp: MLP 保留的层数深度 (例如 22 表示只保留 0-21 层，22层之后剪枝)
    """
    
    print(f"[DOP Setup] Applying MLP Pruning. Keep Layers (dp) = {target_dp}")

    # --- B. MLP 替换 ---
    layers = model.model.layers
    for i, layer in enumerate(layers):
        if not isinstance(layer.mlp, DOPQwenMLP):
            original_mlp = layer.mlp
            layer.mlp = DOPQwenMLP(original_mlp, layer_idx=i, pruning_depth_dp=target_dp)
    
    # --- C. Hook generate ---
    if not hasattr(model.generate, "_is_dop_hooked"):
        original_generate = model.generate

        def dop_generate_wrapper(*args, **kwargs):
            input_ids = kwargs.get('input_ids', None)
            if input_ids is None and len(args) > 0:
                input_ids = args[0]
                
            if input_ids is not None:
                # Qwen2.5-VL 特殊 Token ID
                vision_start = 151652 
                vision_end = 151653
                
                mask = torch.zeros_like(input_ids, dtype=torch.float32)
                
                # 标记视觉区域
                for b in range(input_ids.shape[0]):
                    in_vision = False
                    for i in range(input_ids.shape[1]):
                        token = input_ids[b, i].item()
                        if token == vision_start:
                            in_vision = True
                            mask[b, i] = 1.0
                        elif token == vision_end:
                            in_vision = False
                            mask[b, i] = 1.0
                        elif in_vision:
                            mask[b, i] = 1.0
                
                DopContext.set_mask(mask)
            
            try:
                return original_generate(*args, **kwargs)
            finally:
                DopContext.set_mask(None)

        dop_generate_wrapper._is_dop_hooked = True
        model.generate = dop_generate_wrapper
        
    return model

# --- 主程序 ---
# 设置数据路径
os.environ['LMUData'] = '/data/zengyq/LMUData' 
os.makedirs(os.environ['LMUData'], exist_ok=True)

from vlmeval.config import supported_VLM
from vlmeval.dataset import build_dataset
from vlmeval.inference import infer_data_job
from vlmeval.smp import get_pred_file_path

try:
    from vlmeval.smp import get_logger
except ImportError:
    import logging
    def get_logger(name):
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        logger.addHandler(handler)
        return logger

def main():
    logger = get_logger('DOP_Eval_PruningOnly')
    
    # 1. 定义要跑的 Benchmark
    benchmarks = [
        "TextVQA_VAL", 
        "ChartQA_TEST", 
        "OCRBench",
        "MMBench_dev_en",
        "MMBench_dev_cn"
    ]
    
    # 2. 定义实验配置 (基于 7B 模型 28 层计算)
    # 假设 Qwen2.5-VL-7B 总层数 L=28
    experiments = {
        # 对照组：无剪枝
        # "original_baseline": {"dp": None},
        
        # 实验组 1: 剩余 80% FLOPs 
        "retain_80_flops": {"dp": 22},
        
        # 实验组 2: 剩余 70% FLOPs
        "retain_70_flops": {"dp": 18},
    }
    
    # 3. 选择模型
    model_name = "Qwen2.5-VL-7B-Instruct" 
    model_path = "/data/zengyq/model/Qwen2.5-VL-7B-Instruct" 

    if model_name not in supported_VLM:
        logger.warning(f"{model_name} not found in supported_VLM. Please check config.")

    # 循环执行实验
    for exp_name, cfg in experiments.items():
        logger.info(f"\n{'='*20} Running {exp_name} {'='*20}")
        
        pred_root = f"results/{exp_name}"
        os.makedirs(pred_root, exist_ok=True)

        try:
            model_cls = supported_VLM[model_name]
            # 实例化模型
            model_wrapper = model_cls(model_path=model_path, root=pred_root)
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            continue

        # --- B. 注入 Patch (仅剪枝) ---
        target_dp = cfg.get('dp')

        if target_dp is not None:
            logger.info(f"Applying MLP Pruning: Keep Layers 0-{target_dp-1} (Pruning depth={target_dp})")
            
            apply_dop_patch(
                model=model_wrapper.model, 
                target_dp=target_dp
            )
        else:
            logger.info("Running Baseline: No pruning applied.")
        
        # --- C. 执行评测 ---
        for dataset_name in benchmarks:
            logger.info(f"Evaluating on {dataset_name}...")
            try:
                dataset = build_dataset(dataset_name)
                if dataset is None:
                    logger.error(f"Dataset {dataset_name} build failed.")
                    continue

                infer_data_job(
                    model=model_wrapper,
                    work_dir=pred_root,
                    model_name=model_name,
                    dataset=dataset,
                    verbose=True,
                    api_nproc=2
                )
            except Exception as e:
                logger.error(f"Failed on {dataset_name}: {e}")
                import traceback
                traceback.print_exc()

            print(f"Starting evaluation calculation...")
            try:
                result_file = get_pred_file_path(pred_root, model_name, dataset_name)
                if os.path.exists(result_file):
                    dataset.evaluate(result_file)
                    print(f"Evaluation finished! Results in {pred_root}")
                else:
                    print(f"Result file missing: {result_file}")
            except Exception as e:
                print(f"Evaluation calculation failed: {e}")

        # 清理显存
        del model_wrapper
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()