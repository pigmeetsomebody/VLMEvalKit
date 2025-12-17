import torch.utils.data.dataloader
from transformers import AutoModelForImageTextToText, AutoProcessor
import os
import sys
import torch.nn.functional as F
import torch
import json
from tqdm import tqdm
import numpy as np
import argparse
import random
from functools import partial
import copy
from qwen_vl_utils import process_vision_info

# ... [Retain ds_collections config] ...
ds_collections = {
    'vqav2_val': {
        'train': 'data/vqav2/vqav2_train.jsonl',
        'test': 'data/vqav2/vqav2_train.jsonl',
        'question': 'data/vqav2/v2_OpenEnded_mscoco_val2014_questions.json',
        'annotation': 'data/vqav2/v2_mscoco_val2014_annotations.json',
        'metric': 'vqa_score',
        'max_new_tokens': 10,
    },
    'vqav2_testdev': {
        'train': 'data/vqav2/vqav2_train.jsonl',
        'test': 'data/vqav2/vqav2_train.jsonl',
        'metric': None,
        'max_new_tokens': 10,
    },
    'okvqa_val': {
        'train': 'data/okvqa/okvqa_train.jsonl',
        'test': 'data/okvqa/okvqa_train.jsonl',
        'question': 'data/okvqa/OpenEnded_mscoco_val2014_questions.json',
        'annotation': 'data/okvqa/mscoco_val2014_annotations.json',
        'metric': 'vqa_score',
        'max_new_tokens': 10,
    },
    'textvqa_val': {
        'train': 'data/textvqa/textvqa_train.jsonl',
        'test': 'data/textvqa/textvqa_train.jsonl',
        'question': 'data/textvqa/textvqa_val_questions.json',
        'annotation': 'data/textvqa/textvqa_val_annotations.json',
        'metric': 'vqa_score',
        'max_new_tokens': 10,
    },
    'vizwiz_val': {
        'train': 'data/vizwiz/vizwiz_train.jsonl',
        'test': 'data/vizwiz/vizwiz_train.jsonl',
        'question': 'data/vizwiz/vizwiz_val_questions.json',
        'annotation': 'data/vizwiz/vizwiz_val_annotations.json',
        'metric': 'vqa_score',
        'max_new_tokens': 10,
    },
    'vizwiz_test': {
        'train': 'data/vizwiz/vizwiz_train.jsonl',
        'test': 'data/vizwiz/vizwiz_test.jsonl',
        'metric': None,
        'max_new_tokens': 10,
    },
    'docvqa_val': {
        'train': 'data/docvqa/train.jsonl',
        'test': 'data/docvqa/val.jsonl',
        'annotation': 'data/docvqa/val/val_v1.0.json',
        'metric': 'anls',
        'max_new_tokens': 100,
    },
    'docvqa_test': {
        'train': 'data/docvqa/train.jsonl',
        'test': 'data/docvqa/test.jsonl',
        'metric': None,
        'max_new_tokens': 100,
    },
    'chartqa_test_human': {
        'train': 'data/chartqa/train_human.jsonl',
        'test': 'data/chartqa/test_human.jsonl',
        'metric': 'relaxed_accuracy',
        'max_new_tokens': 100,
    },
    'chartqa_test_augmented': {
        'train': 'data/chartqa/train_augmented.jsonl',
        'test': 'data/chartqa/test_augmented.jsonl',
        'metric': 'relaxed_accuracy',
        'max_new_tokens': 100,
    },
    'gqa_testdev': {
        'train': 'data/gqa/train.jsonl',
        'test': 'data/gqa/testdev_balanced.jsonl',
        'metric': 'accuracy',
        'max_new_tokens': 10,
    },
    'ocrvqa_val': {
        'train': 'data/ocrvqa/ocrvqa_train.jsonl',
        'test': 'data/ocrvqa/ocrvqa_val.jsonl',
        'metric': 'accuracy',
        'max_new_tokens': 100,
    },
    'ocrvqa_test': {
        'train': 'data/ocrvqa/ocrvqa_train.jsonl',
        'test': 'data/ocrvqa/ocrvqa_test.jsonl',
        'metric': 'accuracy',
        'max_new_tokens': 100,
    },
    'ai2diagram_test': {
        'train': 'data/ai2diagram/train.jsonl',
        'test': 'data/ai2diagram/test.jsonl',
        'metric': 'accuracy',
        'max_new_tokens': 10,
    }
}

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(script_dir))))
sys.path.insert(0, str(parent_dir))

# --- Scalar Calculation ---
def compute_optimal_scalar(ref_hiddens, pruned_hiddens):
    """
    Computes scalar 'a' to minimize || H_ref - a * H_pruned ||^2
    """
    # Flatten everything to 1D vectors
    ref_flat = ref_hiddens.reshape(-1).double()
    pruned_flat = pruned_hiddens.reshape(-1).double()
    
    # Align lengths if necessary
    min_len = min(ref_flat.shape[0], pruned_flat.shape[0])
    ref_flat = ref_flat[:min_len]
    pruned_flat = pruned_flat[:min_len]
    
    dot_prod = torch.dot(ref_flat, pruned_flat)
    norm_sq = torch.dot(pruned_flat, pruned_flat)
    
    if norm_sq == 0:
        return 1.0
        
    alpha = (dot_prod / norm_sq).float().item()
    return alpha

# --- Hook Management ---
_hooks = []

def clear_hooks():
    global _hooks
    for h in _hooks:
        h.remove()
    _hooks = []

def register_scaling_hook(model, layer_idx, scalar):
    def scale_output_hook(module, args, output):
        if isinstance(output, tuple):
            return (output[0] * scalar,) + output[1:]
        return output * scalar

    # Qwen2.5-VL specific path
    layer = model.model.language_model.layers[layer_idx]
    handle = layer.register_forward_hook(scale_output_hook)
    _hooks.append(handle)

def set_text_model_config_skip_layer_and_layer_scalar(model, skip_mlp_list, skip_attn_list, mlp_scalars):
    # Ensure lists are unique and sorted for consistency
    model.model.language_model.skip_layer_mlp = sorted(list(set(skip_mlp_list)))
    model.model.language_model.skip_layer_attn = sorted(list(set(skip_attn_list)))
    
    clear_hooks()
    # Apply scalars only for MLPs that are actually skipped
    for layer_idx, scalar in mlp_scalars.items():
        if layer_idx in skip_mlp_list:
            register_scaling_hook(model, layer_idx, scalar)

def extract_image_hidden_states(hidden_states, image_masks):
    """
    hidden_states: [Batch, Seq_Len, Dim]
    image_masks: [Batch, Seq_Len] (False = Image Token)
    Returns: [Total_Image_Tokens, Dim]
    """
    # Create mask: True for Image Tokens
    selection_mask = ~image_masks.to(hidden_states.device)
    
    # Validate shapes
    if hidden_states.shape[:2] != selection_mask.shape:
        # If shapes don't match (e.g. generation steps), take min length
        min_seq = min(hidden_states.shape[1], selection_mask.shape[1])
        hidden_states = hidden_states[:, :min_seq]
        selection_mask = selection_mask[:, :min_seq]

    selected_features = hidden_states[selection_mask] 
    return selected_features.detach().cpu() # Move to CPU to save GPU memory

def get_logits_and_hiddens(model, inputs, image_masks, n_generation_tokens):
    """
    Returns:
    - Logits: Tensor
    - Hidden States: List of Tensors (one per layer), each containing aggregated image token states
    """
    model.eval()
    logits_all = []
    
    with torch.no_grad():
        outputs = model(**inputs, use_cache=True, kwargs={"image_masks": image_masks}, output_hidden_states=True)
        logits = outputs.logits[:, -1, :]
        past = outputs.past_key_values
        logits_all.append(logits.detach().cpu())
        next_token = torch.argmax(logits, dim=-1)
        
        # Extract hidden states only for the Prompt phase (image tokens usually don't appear in generation)
        # Outputs.hidden_states is a tuple of (Layer 0, Layer 1, ... Layer N)
        # Each element is [Batch, Seq, Dim]
        current_layer_hiddens = [
            extract_image_hidden_states(h, image_masks) for h in outputs.hidden_states
        ]
        
        # Generation Loop (We generally ignore hidden states here as they are text tokens)
        for _ in range(1, n_generation_tokens):
            outputs = model(
                input_ids=next_token.unsqueeze(-1),
                past_key_values=past,
                use_cache=True,
                kwargs={"image_masks": image_masks}
            )
            logits = outputs.logits[:, -1, :]
            past = outputs.past_key_values
            logits_all.append(logits.detach().cpu())
            next_token = torch.argmax(logits, dim=-1)
            
    return torch.concatenate(logits_all), current_layer_hiddens

def get_batch_logits_and_hiddens(model, processor, dataloader, n_samples, n_generation_tokens):
    batch_logits = []
    
    # Store aggregated hidden states for every layer
    # Structure: List of Lists -> [Layer_Idx][Batch_Idx] -> Tensor
    layers_aggregator = [[] for _ in range(model.config.num_hidden_layers + 1)] # +1 for embedding output if included
    
    for i, (_, inputs, _) in enumerate(dataloader):
        if i >= n_samples: break
        
        image_masks = get_image_token_masks(model, inputs)
        inputs = inputs.to(model.device)
        
        logits, layer_hiddens_batch = get_logits_and_hiddens(model, inputs, image_masks, n_generation_tokens)
        
        batch_logits.append(logits)
        
        # Aggregate hidden states per layer
        for layer_idx, h_tensor in enumerate(layer_hiddens_batch):
            if layer_idx < len(layers_aggregator):
                layers_aggregator[layer_idx].append(h_tensor)
        
        del inputs
        torch.cuda.empty_cache()
    
    # Concatenate all batches for each layer
    # Result: List of [Total_Image_Tokens, Dim] tensors
    final_layer_hiddens = [torch.cat(batches, dim=0) if batches else torch.empty(0) for batches in layers_aggregator]
    
    return torch.concatenate(batch_logits), final_layer_hiddens

# ... [InferenceSampler, collate_fn, get_image_token_masks, VQADataset] ...
# (Assuming these are correct as per previous code)
def get_image_token_masks(model, inputs):
    bsz, seq_len = inputs.input_ids.shape
    image_masks_list = []
    for i in range(bsz):
        image_token_mask = torch.ones(seq_len, dtype=torch.bool)
        try:
            image_index = torch.where(inputs.input_ids[i] == model.config.image_token_id)[0]
            image_token_mask[image_index] = False
        except: pass
        try:
            vision_index = torch.where(inputs.input_ids[i] == model.config.vision_token_id)[0]
            image_token_mask[vision_index] = False
        except: pass
        image_masks_list.append(image_token_mask)
    return torch.stack(image_masks_list, dim=0)

class InferenceSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, size):
        self._size = int(size)
        self._local_indices = range(0, size) 
    def __iter__(self):
        yield from self._local_indices
    def __len__(self):
        return len(self._local_indices)

def collate_fn(batches, processor):
    questions = [_['question'] for _ in batches]
    question_ids = [_['question_id'] for _ in batches]
    batch_messages = []
    for batch in batches:
        messages = []
        for shot in batch['few_shot_samples']:
            messages.append({
                "role": "user",
                "content": [{"type": "image", "image": shot['image']}, {"type": "text", "text": shot['question']}],
            })
            messages.append({"role": "assistant", "content": shot['answer']})
        messages.append({
            "role": "user",
            "content": [{"type": "image", "image": batch['image']}, {"type": "text", "text": batch["question"]}],
        })
        batch_messages = messages
    
    text = processor.apply_chat_template(batch_messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(batch_messages)
    inputs = processor(
        text=text,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
        padding_side='left',
    )
    return question_ids, inputs, batch_messages

class VQADataset(torch.utils.data.Dataset):
    def __init__(self, train, test, prompt, few_shot):
        self.test = open(test).readlines()
        self.prompt = prompt
        self.few_shot = few_shot
        if few_shot > 0:
            self.train = open(train).readlines()
    def __len__(self):
        return len(self.test)
    def __getitem__(self, idx):
        data = json.loads(self.test[idx].strip())
        return {
            "few_shot_samples": [],
            "few_shot_prompt": self.prompt.format("<|image_pad|>", data['question']),
            "image": data['image'],
            'question': data['question'],
            'question_id': data['question_id'],
            'annotation': data.get('answer', None)
        }

# --- Core Logic ---

def calculate_kl_score_and_new_scalar(model, processor, dataloader, n_samples, n_generation_tokens, 
                                      gold_logits, gold_layer_hiddens, 
                                      skip_mlp, skip_attn, mlp_scalars, target_layer_idx=None):
    """
    1. Applies Config.
    2. Runs Inference.
    3. If target_layer_idx is provided, calculates scalar ONLY for that layer.
    4. Computes KL Score.
    """
    # Apply Config (using existing scalars for other layers)
    set_text_model_config_skip_layer_and_layer_scalar(model, skip_mlp, skip_attn, mlp_scalars)
    
    # Run Inference
    curr_logits, curr_layer_hiddens = get_batch_logits_and_hiddens(model, processor, dataloader, n_samples, n_generation_tokens)
    
    computed_scalar = None
    if target_layer_idx is not None:
        # Note: hidden_states[i] usually corresponds to output of layer i-1 or input to layer i depending on architecture.
        # Transformers usually outputs (embeddings, layer_0_out, layer_1_out...)
        # If we prune layer L, we want to align the output of Layer L.
        # Indexing: gold_layer_hiddens[L+1] is roughly the output of Layer L block.
        # We safer logic: Use index L+1 for layer L output.
        h_idx = target_layer_idx + 1
        if h_idx < len(gold_layer_hiddens) and h_idx < len(curr_layer_hiddens):
            computed_scalar = compute_optimal_scalar(gold_layer_hiddens[h_idx], curr_layer_hiddens[h_idx])
        else:
            computed_scalar = 1.0

    # Align Lengths for KL
    min_len = min(curr_logits.shape[0], gold_logits.shape[0])
    p_log_prob = F.log_softmax(gold_logits[:min_len], dim=-1)
    q_log_prob = F.log_softmax(curr_logits[:min_len], dim=-1)
    
    kl = F.kl_div(q_log_prob, p_log_prob, reduction='batchmean', log_target=True)
    return kl.item(), computed_scalar

def run_mixed_beam_search(model, processor, dataloader, n_samples, n_generation_tokens, 
                          all_mlp_candidates, all_attn_candidates, target_saving_benefits, beam_width=3):
    
    BENEFIT_MLP = 1.0
    BENEFIT_ATTN = 0.5

   

    initial_beam = {
        'skip_mlp': [],
        'skip_attn': [],
        'scalars': {}, # Dict: {layer_idx: value}
        'rem_mlp': list(all_mlp_candidates),
        'rem_attn': list(all_attn_candidates),
        'score': 0.0,
        'current_benefit': 0.0,
    }
    
    active_beams = [initial_beam]
    completed_beams = []
    
    step_counter = 0

    while active_beams:
        step_counter += 1
        print(f"\n=== Search Step {step_counter} (Active: {len(active_beams)}) ===")
        all_candidates = []
        
        for beam in active_beams:
             # 1. Calculate Gold Baselines
            print("Calculating Gold Logits & Hiddens...")
            print(f"current beam: {beam}")
    
            set_text_model_config_skip_layer_and_layer_scalar(model, beam['skip_mlp'], beam['skip_attn'], beam['scalars'].copy())

            gold_logits, gold_layer_hiddens = get_batch_logits_and_hiddens(model, processor, dataloader, n_samples, n_generation_tokens)
            print("Gold Baselines Ready.")
            
            # --- Option A: MLP Pruning (With Scalar Calculation) ---
            for mlp_cand in beam['rem_mlp']:
                new_benefit = beam['current_benefit'] + BENEFIT_MLP
                new_skip_mlp = beam['skip_mlp'] + [mlp_cand]
                new_rem_mlp = [x for x in beam['rem_mlp'] if x != mlp_cand]
                
                # We do NOT add the new scalar yet, we pass it as target to calculate it
                temp_scalars = beam['scalars'].copy() 
                
                # First Pass: Calculate Scalar & Score
                # We prune 'mlp_cand', but haven't computed its scalar yet.
                # The function will compute scalar for 'mlp_cand' based on current run.
                score_no_scalar, new_scalar_val = calculate_kl_score_and_new_scalar(
                    model, processor, dataloader, n_samples, n_generation_tokens, 
                    gold_logits, gold_layer_hiddens, 
                    new_skip_mlp, beam['skip_attn'], temp_scalars, 
                    target_layer_idx=mlp_cand
                )
                
                # Second Pass (Optional/Implicit): If you want exact score WITH the new scalar, 
                # you technically need to run again. 
                # For greedy speed, we can assume applying the scalar improves/maintains the approximation.
                # Let's verify with the scalar applied.
                temp_scalars[mlp_cand] = new_scalar_val
                
                # final_score, _ = calculate_kl_score_and_new_scalar(
                #     model, processor, dataloader, n_samples, n_generation_tokens, 
                #     gold_logits, gold_layer_hiddens, 
                #     new_skip_mlp, beam['skip_attn'], temp_scalars, 
                #     target_layer_idx=None # No new scalar needed
                # )
                
                all_candidates.append({
                    'type': 'mlp',
                    'added_layer': mlp_cand,
                    'skip_mlp': new_skip_mlp,
                    'skip_attn': beam['skip_attn'],
                    'rem_mlp': new_rem_mlp,
                    'rem_attn': beam['rem_attn'],
                    'scalars': temp_scalars,
                    'score': score_no_scalar,
                    'current_benefit': new_benefit
                })
                print(f"  MLP Candidate: {all_candidates[-1]}")

            # --- Option B: Attention Pruning ---
            for attn_cand in beam['rem_attn']:
                new_benefit = beam['current_benefit'] + BENEFIT_ATTN
                new_skip_attn = beam['skip_attn'] + [attn_cand]
                new_rem_attn = [x for x in beam['rem_attn'] if x != attn_cand]
                
                score, _ = calculate_kl_score_and_new_scalar(
                    model, processor, dataloader, n_samples, n_generation_tokens, 
                    gold_logits, gold_layer_hiddens, 
                    beam['skip_mlp'], new_skip_attn, beam['scalars'],
                    target_layer_idx=None
                )
                
                all_candidates.append({
                    'type': 'attn',
                    'added_layer': attn_cand,
                    'skip_mlp': beam['skip_mlp'],
                    'skip_attn': new_skip_attn,
                    'rem_mlp': beam['rem_mlp'],
                    'rem_attn': new_rem_attn,
                    'scalars': beam['scalars'].copy(),
                    'score': score,
                    'current_benefit': new_benefit
                })


        if not all_candidates:
            break

        # Select Top K
        all_candidates.sort(key=lambda x: x['score'])
        best_candidates = all_candidates[:beam_width]
        
        print(f"  Best Score: {best_candidates[0]['score']:.6f}")
        print(f"  Skip MLP: {best_candidates[0]['skip_mlp']}")
        print(f"  Skip MLP scalars: {best_candidates[0]['scalars']}")

        next_active_beams = []
        for cand in best_candidates:
            if cand['current_benefit'] >= target_saving_benefits:
                completed_beams.append(cand)
            else:
                next_active_beams.append(cand)
        
        active_beams = next_active_beams

    # Finalization
    if not completed_beams:
        completed_beams = active_beams
        
    completed_beams.sort(key=lambda x: x['score'])
    return completed_beams

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--checkpoint', type=str, default="/data/share/Qwen2.5-VL-7B-Instruct/")
    parser.add_argument('--batch_size', type=int, default=1) 
    parser.add_argument('--dataset', type=str, default='textvqa_val', choices=list(ds_collections.keys()))
    parser.add_argument('--n-samples', type=int, default=100)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--beam_width', type=int, default=2)
    parser.add_argument('--target_saving_benefits', type=float, default=2.0)
    args = parser.parse_args()

    model = AutoModelForImageTextToText.from_pretrained(args.checkpoint, dtype=torch.bfloat16, device_map="cuda")
    processor = AutoProcessor.from_pretrained(args.checkpoint, trust_remote_code=True)
    
    prompt = '<|vision_start|>{}<|vision_end|> {} Answer:'
    random.seed(args.seed)
    dataset = VQADataset(
        train=ds_collections[args.dataset]['train'],
        test=ds_collections[args.dataset]['test'],
        prompt=prompt,
        few_shot=0,
    )
    
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        sampler=InferenceSampler(size=args.n_samples),
        batch_size=args.batch_size,
        collate_fn=partial(collate_fn, processor=processor),
    )

    # Example: Prune from layer 24 to 26
    all_mlp_candidates = [i for i in range(3, model.config.num_hidden_layers)]
    all_attn_candidates = [i for i in range(3, model.config.num_hidden_layers)]

    final_pool = run_mixed_beam_search(
        model, processor, dataloader, args.n_samples, 10,
        all_mlp_candidates, all_attn_candidates, args.target_saving_benefits, args.beam_width
    )

    print("\n=== FINAL POLICY ===")
    best = final_pool[0]
    print(f"Final Score: {best['score']}")
    print(f"Skipped MLPs: {best['skip_mlp']}")
    print(f"Skipped Attns: {best['skip_attn']}")
    print(f"Scalars: {best['scalars']}")

if __name__ == "__main__":
    main()