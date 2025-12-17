import torch
import torch.nn as nn
from transformers.models.qwen2.modeling_qwen2 import Qwen2MLP
import os
import sys
import math

# --- 上下文管理器 (用于传递视觉 Mask) ---
class IMAContext:
    _visual_mask = None
    
    @classmethod
    def set_mask(cls, mask):
        cls._visual_mask = mask
        
    @classmethod
    def get_mask(cls):
        return cls._visual_mask

# --- 核心组件1: 支持 IMA 跳过计算的 MLP ---
class IMAQwenMLP(nn.Module):
    def __init__(self, original_mlp, layer_idx, skip_ratio):
        super().__init__()
        # 复制参数
        self.gate_proj = original_mlp.gate_proj
        self.up_proj = original_mlp.up_proj
        self.down_proj = original_mlp.down_proj
        self.act_fn = original_mlp.act_fn
        
        self.layer_idx = layer_idx
        self.skip_ratio = skip_ratio  # 视觉 Token 被跳过的概率

    def forward(self, x):
        # x shape: [batch, seq_len, hidden_dim]
        
        # 1. 获取视觉 Mask (1 为视觉 Token, 0 为文本 Token)
        # 注意：这需要 Generate Hook 配合注入
        visual_mask = IMAContext.get_mask() 
        
        # 如果没有 Mask 或者不在 Prefill 阶段 (decode阶段通常长度为1)，或者 skip_ratio 为 0，则执行标准计算
        if visual_mask is None or self.skip_ratio <= 0.0:
            return self._standard_forward(x)
            
        # 检查 Mask 形状是否匹配 (防止 decode 阶段出错)
        if visual_mask.shape[1] != x.shape[1]:
            return self._standard_forward(x)

        # 2. 生成跳过 Mask (Bernoulli Sample)
        # 论文方法：Skipping Computations (Fig 11)
        # 只有视觉 Token 需要被考虑跳过
        
        # 将 visual_mask 对齐设备
        visual_mask = visual_mask.to(x.device).to(x.dtype) # [B, L]
        
        # 生成随机概率矩阵 [B, L]
        rand_probs = torch.rand_like(visual_mask)
        
        # 生成 Skip Decision Mask: 
        # 如果是视觉 Token (mask=1) 且 随机值 < skip_ratio，则 skip (置1)，否则保留 (置0)
        # 文本 Token (mask=0) 永远不 skip
        should_skip_mask = (visual_mask > 0.5) & (rand_probs < self.skip_ratio)
        should_skip_mask = should_skip_mask.unsqueeze(-1) # [B, L, 1] 广播用

        # 3. 执行计算 (模拟硬件跳过：计算结果 * (1 - skip_mask))
        # 在真实硬件优化中，这部分矩阵乘法根本不会执行。
        # 在 PyTorch 模拟中，我们计算后置零，以复现精度影响。
        
        gate_out = self.gate_proj(x)
        up_out = self.up_proj(x)
        intermediate = self.act_fn(gate_out) * up_out
        output = self.down_proj(intermediate)
        
        # 应用 Mask: 被跳过的部分置零 (残差连接在外部 Transformer Block 中，所以这里输出 0 等于直接通过残差)
        output = output * (1.0 - should_skip_mask.to(output.dtype))
        
        return output

    def _standard_forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

# --- FLOPs 计算与参数推导工具 ---
def calculate_target_skip_ratio(model_config, target_flops_retention, estimated_visual_ratio=0.5):
    """
    严格计算达到目标 FLOPs 所需的 visual_skip_ratio。
    
    Args:
        model_config: 模型配置
        target_flops_retention: 目标保留比例 (例如 0.8 表示 80% FLOPs)
        estimated_visual_ratio: 估计输入中视觉 Token 的占比 (默认 0.5)
    """
    hidden = model_config.hidden_size
    inter = model_config.intermediate_size
    num_layers = model_config.num_hidden_layers
    
    # 1. 计算单层理论 FLOPs (忽略 SeqLen^2 的 Attention Score 计算，仅看线性部分，因为是主要部分)
    # MLP FLOPs: 3个矩阵 (Gate, Up, Down) -> 3 * (hidden * inter) * 2 (mul+add)
    flops_mlp_per_token = 3 * (hidden * inter) * 2
    
    # Attn FLOPs: 4个矩阵 (Q, K, V, O) -> 4 * (hidden * hidden) * 2
    flops_attn_per_token = 4 * (hidden * hidden) * 2
    
    flops_layer_per_token = flops_mlp_per_token + flops_attn_per_token
    total_flops_per_token = flops_layer_per_token * num_layers
    
    print(f"[FLOPs Analysis] Per Token:")
    print(f"   - MLP FLOPs: {flops_mlp_per_token / 1e6:.2f} M ({flops_mlp_per_token/flops_layer_per_token:.1%})")
    print(f"   - Attn FLOPs: {flops_attn_per_token / 1e6:.2f} M ({flops_attn_per_token/flops_layer_per_token:.1%})")
    
    # 2. 建立方程求解 Skip Ratio (S)
    # Target_FLOPs = Total * Retention
    # Target_FLOPs = (All_Attn) + (Text_MLP) + (Vis_MLP_Kept)
    #              = (N * Attn) + (N * (1-v) * MLP) + (N * v * (1-S) * MLP)
    # 其中 N=1 (per token), v = visual_ratio
    
    # Retention * (Attn + MLP) = Attn + (1-v)*MLP + v*(1-S)*MLP
    # Retention * Attn + Retention * MLP = Attn + MLP - v*MLP + v*MLP - v*S*MLP
    # Retention * Attn + Retention * MLP = Attn + MLP - v*S*MLP
    # v*S*MLP = Attn + MLP - (Retention * Attn + Retention * MLP)
    # v*S*MLP = (1 - Retention) * (Attn + MLP)
    # S = [(1 - Retention) * Total_FLOPs] / [v * MLP_FLOPs]
    
    mlp_ratio = flops_mlp_per_token / flops_layer_per_token
    
    if estimated_visual_ratio <= 0:
        return 0.0

    # 核心公式
    required_reduction = 1.0 - target_flops_retention
    # 我们需要通过砍掉一部分 Visual MLP 来达到这个 Reduction
    # Visual MLP 占总计算量的比例 = visual_ratio * mlp_ratio
    max_possible_reduction = estimated_visual_ratio * mlp_ratio
    
    if required_reduction > max_possible_reduction:
        print(f"Warning: Target {target_flops_retention*100}% FLOPs is impossible.")
        print(f"   Even skipping 100% visual MLP only reduces {max_possible_reduction*100:.2f}%.")
        return 1.0
        
    skip_ratio = required_reduction / (estimated_visual_ratio * mlp_ratio)
    
    print(f"[Target Calculation]")
    print(f"   - Target Retention: {target_flops_retention*100}%")
    print(f"   - Estimated Vis Ratio: {estimated_visual_ratio*100}%")
    print(f"   - Calculated Visual Skip Ratio: {skip_ratio:.4f} ({skip_ratio*100:.2f}%)")
    
    return skip_ratio

# --- 核心组件2: 模型注入工具 ---
def apply_ima_patch(model, target_retention, estimated_vis_ratio=0.5):
    """
    target_retention: 目标 FLOPs 保留率 (e.g., 0.8)
    estimated_vis_ratio: 估计的视觉 token 占比 (用于计算 skip_ratio)
    """
    
    # 1. 计算需要的 skip_ratio
    if target_retention >= 1.0:
        calculated_skip_ratio = 0.0
    else:
        calculated_skip_ratio = calculate_target_skip_ratio(
            model.config, target_retention, estimated_vis_ratio
        )
    
    print(f"[IMA Setup] Applying FFN Skipping. Ratio = {calculated_skip_ratio:.4f}")

    # 2. MLP 替换
    layers = model.model.layers
    # 论文中提到 Skipping 通常从中间层开始 (start layer)，或者全层应用
    # 这里为了严格满足 FLOPs 限制，我们全层应用，或者你可以加一个 start_layer 参数
    for i, layer in enumerate(layers):
        if not isinstance(layer.mlp, IMAQwenMLP):
            original_mlp = layer.mlp
            layer.mlp = IMAQwenMLP(original_mlp, layer_idx=i, skip_ratio=calculated_skip_ratio)
        else:
            # 如果已经是 IMAMLP，只需更新 ratio
            layer.mlp.skip_ratio = calculated_skip_ratio
            
    # 3. Hook generate (保持不变，用于识别视觉 Token)
    if not hasattr(model.generate, "_is_dop_hooked"):
        original_generate = model.generate

        def dop_generate_wrapper(*args, **kwargs):
            input_ids = kwargs.get('input_ids', None)
            if input_ids is None and len(args) > 0:
                input_ids = args[0]
            
            # 安全获取 vision tokens id (兼容 Qwen2-VL)
            vision_start = getattr(model.config, "vision_start_token_id", 151652)
            vision_end = getattr(model.config, "vision_end_token_id", 151653)
                
            if input_ids is not None:
                mask = torch.zeros_like(input_ids, dtype=torch.float32)
                # 标记视觉区域
                # 简单逻辑：在 vision_start 和 vision_end 之间的即为视觉 token
                # 注意：这里假设 batch 中每个样本逻辑一致，简单循环处理
                cpu_ids = input_ids.cpu()
                for b in range(cpu_ids.shape[0]):
                    in_vision = False
                    for i in range(cpu_ids.shape[1]):
                        token = cpu_ids[b, i].item()
                        if token == vision_start:
                            in_vision = True
                            mask[b, i] = 1.0 # start token 也算
                        elif token == vision_end:
                            in_vision = False
                            mask[b, i] = 1.0 # end token 也算
                        elif in_vision:
                            mask[b, i] = 1.0
                
                IMAContext.set_mask(mask)
            
            try:
                return original_generate(*args, **kwargs)
            finally:
                IMAContext.set_mask(None)

        dop_generate_wrapper._is_dop_hooked = True
        model.generate = dop_generate_wrapper
        
    return model

# --- 主程序 ---
os.environ['LMUData'] = '/data/zengyq/LMUData' 
os.makedirs(os.environ['LMUData'], exist_ok=True)

from vlmeval.config import supported_VLM
from vlmeval.dataset import build_dataset
from vlmeval.inference import infer_data_job
from vlmeval.smp import get_pred_file_path, get_logger

def main():
    logger = get_logger('IMA_Eval_FFN_Skipping')
    
    benchmarks = [
        "TextVQA_VAL", 
        "ChartQA_TEST", 
        "OCRBench",
        "MMBench_dev_en",
        "MMBench_dev_cn"
    ]
    
    # 定义实验: 严格按 FLOPs 保留率
    # 注意：estimated_vis_ratio 是必须的假设，因为不知道具体图片的 token 数
    # Qwen2-VL 图片 token 是动态的，通常一张 1024 图约 1000 token。加上 system prompt，0.5~0.6 是合理估计。
    ESTIMATED_VIS_RATIO = 0.55 
    
    experiments = {
        # "baseline_100": {"flops": 1.0},
        "prune_retain_80": {"flops": 0.80},
        "prune_retain_70": {"flops": 0.70},
    }
    
    model_name = "Qwen2.5-VL-7B-Instruct" 
    model_path = "/data/zengyq/model/Qwen2.5-VL-7B-Instruct" 

    if model_name not in supported_VLM:
        logger.warning(f"{model_name} not found in supported_VLM.")

    for exp_name, cfg in experiments.items():
        logger.info(f"\n{'='*20} Running {exp_name} {'='*20}")
        
        pred_root = f"results/{exp_name}"
        os.makedirs(pred_root, exist_ok=True)

        try:
            model_cls = supported_VLM[model_name]
            model_wrapper = model_cls(model_path=model_path, root=pred_root)
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            continue

        # --- 应用 IMA Patch ---
        target_flops = cfg.get('flops')
        apply_ima_patch(
            model=model_wrapper.model, 
            target_retention=target_flops,
            estimated_vis_ratio=ESTIMATED_VIS_RATIO
        )
        
        # --- 执行评测 ---
        for dataset_name in benchmarks:
            logger.info(f"Evaluating on {dataset_name}...")
            try:
                dataset = build_dataset(dataset_name)
                infer_data_job(
                    model=model_wrapper,
                    work_dir=pred_root,
                    model_name=model_name,
                    dataset=dataset,
                    verbose=True,
                    api_nproc=1
                )
            except Exception as e:
                logger.error(f"Failed on {dataset_name}: {e}")
                import traceback
                traceback.print_exc()
            # 结果计算
            print(f"Starting evaluation calculation...")
            try:
                result_file = get_pred_file_path(pred_root, model_name, dataset_name)
                if os.path.exists(result_file):
                    dataset.evaluate(result_file)
                    print(f"Evaluation finished! Results in {pred_root}")
                else:
                    print(f"Result file missing: {result_file}")
            except Exception as e:
                print(f" Evaluation calculation failed: {e}")

        del model_wrapper
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()