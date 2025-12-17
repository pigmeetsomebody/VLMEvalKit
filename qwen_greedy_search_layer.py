import torch.utils.data.dataloader
from transformers import AutoModelForCausalLM, AutoTokenizer
from qwen_vl_utils import process_vision_info
from transformers import AutoModelForImageTextToText, AutoProcessor
from transformers.generation import GenerationConfig
import os
import sys
import re
import torch.nn.functional as F
import torch
import torch.nn as nn
import json
from tqdm import tqdm
import numpy as np
import pandas as pd
import json
import argparse
import itertools
import random
from functools import partial

ds_collections = {
    'vqav2_val': {
        'train': 'data/vqav2/vqav2_train.jsonl',
        'test': 'data/vqav2/vqav2_val.jsonl',
        'question': 'data/vqav2/v2_OpenEnded_mscoco_val2014_questions.json',
        'annotation': 'data/vqav2/v2_mscoco_val2014_annotations.json',
        'metric': 'vqa_score',
        'max_new_tokens': 10,
    },
    'vqav2_testdev': {
        'train': 'data/vqav2/vqav2_train.jsonl',
        'test': 'data/vqav2/vqav2_testdev.jsonl',
        'metric': None,
        'max_new_tokens': 10,
    },
    'okvqa_val': {
        'train': 'data/okvqa/okvqa_train.jsonl',
        'test': 'data/okvqa/okvqa_val.jsonl',
        'question': 'data/okvqa/OpenEnded_mscoco_val2014_questions.json',
        'annotation': 'data/okvqa/mscoco_val2014_annotations.json',
        'metric': 'vqa_score',
        'max_new_tokens': 10,
    },
    'textvqa_val': {
        'train': 'data/textvqa/textvqa_train.jsonl',
        'test': 'data/textvqa/textvqa_val.jsonl',
        'question': 'data/textvqa/textvqa_val_questions.json',
        'annotation': 'data/textvqa/textvqa_val_annotations.json',
        'metric': 'vqa_score',
        'max_new_tokens': 10,
    },
    'vizwiz_val': {
        'train': 'data/vizwiz/vizwiz_train.jsonl',
        'test': 'data/vizwiz/vizwiz_val.jsonl',
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

def set_text_model_config_skip_layer(model, append_block_mlp=None, drop_block_mlp=None, append_block_attn=None, drop_block_attn=None):
    if append_block_mlp:
        model.model.language_model.skip_layer_mlp.append(append_block_mlp)
    if drop_block_mlp:
        model.model.language_model.skip_layer_mlp.remove(drop_block_mlp)
    if append_block_attn:
        model.model.language_model.skip_layer_attn.append(append_block_attn)
    if drop_block_attn:
        model.model.language_model.skip_layer_attn.remove(drop_block_attn)


def get_logits(model, processor, inputs, image_masks, n_generation_tokens):
    model.eval()
    logits_all = []
    # print(f"inputs:\n{inputs}")

    with torch.no_grad():
        outputs = model(**inputs, use_cache=True, kwargs={"image_masks": None})
        logits = outputs.logits[:, -1, :]
        past = outputs.past_key_values
        logits_all.append(logits.detach().cpu())
        next_token = torch.argmax(logits, dim=-1)
        generated_ids = torch.cat([inputs['input_ids'], next_token.unsqueeze(-1)], dim=-1)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
        ]
        output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        for step in range(1, n_generation_tokens):
            outputs = model(
                input_ids=next_token.unsqueeze(-1),
                past_key_values=past,
                use_cache=True,
                kwargs={"image_masks": None}
            )
            logits = outputs.logits[:, -1, :]
            past = outputs.past_key_values
            logits_all.append(logits.detach().cpu())
            next_token = torch.argmax(logits, dim=-1)
            generated_ids = torch.cat([generated_ids, next_token.unsqueeze(-1)], dim=-1)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
            ]
            output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        # print(f"Generated Texts: {output_text}")
    return torch.concatenate(logits_all)


def get_batch_logits(model, processor, dataloader, n_samples, n_generation_tokens):
    batch_logits = []
    for _, (question_ids, inputs, messages) in tqdm(enumerate(dataloader)):
        image_masks = get_image_token_masks(model, inputs)
        # print(f"Processing question IDs: {question_ids}")
        # print(f"messages: {messages}")
        inputs = inputs.to(model.device)
        batch_logits.append(get_logits(model, processor, inputs, image_masks, n_generation_tokens))
        del inputs
        torch.cuda.empty_cache()
    return torch.concatenate(batch_logits)




def greedy_search(model, processor, dataloader, n_samples=20, n_generation_tokens=20, prun_mlp_deque=[], prun_attn_deque=[], gold_skip_mlp=[], gold_skip_attn=[]):
    model.model.language_model.skip_layer_mlp = gold_skip_mlp
    model.model.language_model.skip_layer_attn = gold_skip_attn
    print(f"Current gold skip MLP layers before: {model.model.language_model.skip_layer_mlp}")
    gold_logits = get_batch_logits(model, processor, dataloader, n_samples, n_generation_tokens)
    scores = []
    for mlp_skip in tqdm(prun_mlp_deque):
        print(f"Trying MLP skip layer: {mlp_skip}")
        set_text_model_config_skip_layer(model, append_block_mlp=mlp_skip)
        print(f"Current gold skip MLP layers: {model.model.language_model.skip_layer_mlp}")
        tmp_logits = get_batch_logits(model, processor, dataloader, n_samples, n_generation_tokens)
        print(f"tmp_logits shape: {tmp_logits.shape}, gold_logits shape: {gold_logits.shape}")
        p_log_prob = F.log_softmax(gold_logits, dim=-1)
        q_log_prob = F.log_softmax(tmp_logits, dim=-1)
        kl = F.kl_div(q_log_prob, p_log_prob, reduction='batchmean', log_target=True)

        print("p_log_prob:", p_log_prob.min().item(), p_log_prob.max().item())
        print("q_log_prob:", q_log_prob.min().item(), q_log_prob.max().item())
        print("exp(q_log_prob).sum:", torch.exp(q_log_prob).sum(-1).mean().item())
        print("KL divergence:", kl.item())
        scores.append(kl.item())
        set_text_model_config_skip_layer(model, drop_block_mlp=mlp_skip)
    print(scores)
    idx_min = int(np.argmin(np.array(scores)))
    gold_skip_mlp.append(prun_mlp_deque[idx_min])
    print(f"Selected MLP skip layer: {prun_mlp_deque[idx_min]} with KL score: {scores[idx_min]}")
    prun_mlp_deque.remove(prun_mlp_deque[idx_min])
    model.model.language_model.skip_layer_mlp = gold_skip_mlp

    gold_logits = get_batch_logits(model, processor, dataloader, n_samples, n_generation_tokens)
    scores = []
    for attn_skip in tqdm(prun_attn_deque):
        
        print(f"Trying Attention skip layer: {attn_skip}")
        set_text_model_config_skip_layer(model, append_block_attn=attn_skip)
        print(f"Current gold skip Attn layers: {model.model.language_model.skip_layer_attn}")

        tmp_logits = get_batch_logits(model, processor, dataloader, n_samples, n_generation_tokens)
        p_log_prob = F.log_softmax(gold_logits, dim=-1)
        q_prob = F.softmax(tmp_logits, dim=-1)
        kl = F.kl_div(p_log_prob, q_prob, reduction='batchmean')
        scores.append(kl.item())
        set_text_model_config_skip_layer(model, drop_block_attn=attn_skip)
    print(scores)
    idx_min = int(np.argmin(np.array(scores)))
    gold_skip_attn.append(prun_attn_deque[idx_min])
    print(f"Selected Attention skip layer: {prun_attn_deque[idx_min]} with KL score: {scores[idx_min]}")
    prun_attn_deque.remove(prun_attn_deque[idx_min])
    return prun_mlp_deque, prun_attn_deque, gold_skip_mlp, gold_skip_attn
    


class InferenceSampler(torch.utils.data.sampler.Sampler):

    def __init__(self, size):
        self._size = int(size)
        assert size > 0
        self._rank = 0
        self._world_size = 4
        self._local_indices = self._get_local_indices(size, self._world_size,
                                                      self._rank)

    @staticmethod
    def _get_local_indices(total_size, world_size, rank):
        shard_size = total_size // world_size
        left = total_size % world_size
        shard_sizes = [shard_size + int(r < left) for r in range(world_size)]

        begin = sum(shard_sizes[:rank])
        end = min(sum(shard_sizes[:rank + 1]), total_size)
        return range(begin, end)

    def __iter__(self):
        yield from self._local_indices

    def __len__(self):
        return len(self._local_indices)

def collate_fn(batches, processor):

    questions = [_['question'] for _ in batches]
    question_ids = [_['question_id'] for _ in batches]
    annotations = [_['annotation'] for _ in batches]
    batch_messages = []
    for batch in batches:
        messages = []
        # print(f"batch: {batch}")
        
        for shot in batch['few_shot_samples']:
            messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": shot['image'],
                    },
                    {
                        "type": "text",
                        "text": shot['question'],
                    },
                ],
            })
            messages.append({
                "role": "assistant",
                "content": shot['answer'],
            })
        messages.append({
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": batch['image'],
                },
                {
                    "type": "text",
                        "text": batch["question"],
                },
            ],
        })
        batch_messages = messages
    # text = [batch["few_shot_prompt"] for batch in batches]
    text = processor.apply_chat_template(batch_messages, tokenize=False, add_generation_prompt=True)
    # print(f"text: {text}")
    
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

def get_image_token_masks(model, inputs):
    bsz, seq_len = inputs.input_ids.shape
    # print(f"input_ids: {inputs.input_ids.shape}\n, {inputs.input_ids}")
    image_masks_list = []
    for i in range(bsz):
        image_token_mask = torch.ones(seq_len, dtype=torch.bool)
        image_index = torch.where(inputs.input_ids[i] == model.config.image_token_id)[0]
        vision_index = torch.where(inputs.input_ids[i] == model.config.vision_token_id)[0]
        image_token_mask[image_index] = False
        image_token_mask[vision_index] = False
        image_masks_list.append(image_token_mask)
    image_token_masks = torch.stack(image_masks_list, dim=0)
    return image_token_masks

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
        image, question, question_id, annotation = data['image'], data[
            'question'], data['question_id'], data.get('answer', None)
        filtered_few_shot_samples = []
        few_shot_prompt = ''
        # if self.few_shot > 0:
        #     few_shot_samples = random.sample(self.train, self.few_shot)
        #     for sample in few_shot_samples:
        #         sample = json.loads(sample.strip())
        #         if question_id == sample['question_id']:
        #             continue
        #         few_shot_prompt += self.prompt.format(
        #             "<|image_pad|>") + f" {sample['answer']}"
        #         filtered_few_shot_samples.append(sample)

        return {
            "few_shot_samples": filtered_few_shot_samples,
            "few_shot_prompt": self.prompt.format("<|image_pad|>", question) + few_shot_prompt,
            "image": image,
            'question': question,
            'question_id': question_id,
            'annotation': annotation
        }
    

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42, help='Random seed for sampling')
    parser.add_argument('--checkpoint', type=str, default="/data/share/Qwen2-VL-7B-Instruct/")
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--dataset', type=str, default='vqav2_val', choices=list(ds_collections.keys()))
    parser.add_argument('--n-samples', type=int, default=100, help='Number of samples for inference')
    parser.add_argument('--num_workers', type=int, default=1, help='Number of workers for data loading')
    args = parser.parse_args()
    model = AutoModelForImageTextToText.from_pretrained(
        args.checkpoint, dtype=torch.bfloat16, device_map="cuda"
    )

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
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=partial(collate_fn, processor=processor),
    )

    # with open(dataset_path, 'r', encoding='utf-8') as file:
    #     for line in tqdm(file):
    #         data = json.loads(line.strip())
    #         image_path = data['image']
    #         question = data['question']
    #         results.append({'image': image_path, 'text': question})
    #         i += 1
    #         if i >= n_samples:
    #             break

    # results += [{
    #     'image': '/home/zhuyy/Qwen-VL/assets/demo.jpeg',
    #     'text': 'Describe this image.',
    # }]
    n_generation_tokens = 10
    target_prun_layer = 5
    # prun_mlp_deque = [25, 5, 20, 4, 19, 27, 15, 16, 22, 23]
    prun_mlp_deque = [i for i in range(3, model.config.num_hidden_layers)]
    prun_attn_deque = [i for i in range(3, model.config.num_hidden_layers)]
    gold_skip_mlp = []
    gold_skip_attn = []
    for i in range(target_prun_layer):
        with torch.no_grad():
            if prun_mlp_deque and prun_attn_deque:
                prun_mlp_deque, prun_attn_deque, gold_skip_mlp, gold_skip_attn = greedy_search(model, processor, dataloader, args.n_samples, n_generation_tokens, prun_mlp_deque, prun_attn_deque, gold_skip_mlp, gold_skip_attn)
            else:
                break
    print("Final selected MLP skip layers:", gold_skip_mlp)
    print("Final selected Attention skip layers:", gold_skip_attn)

if __name__ == "__main__":
    main()