import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import argparse
import sys
import os
import random
from tqdm import tqdm
from utils import govt_name

# --- 1. PROMPT MANAGERS ---

class GeneratePrompter:
    """Handles prompts for generating NEW essays."""
    def __init__(self):
        self.system_msg= ("You are a middle school student who writes essays.")
        self.persuasive_instruction = (
            "Please read the below prompt and write an essay: "
        )
        self.source_instruction = (
            "Please read the source essay and write an essay based on given topic prompt."
        )

    def get_tasks(self, topic, n_essays):
        """Returns a list of task dicts for generating new essays for this topic."""
        if topic["type"]=="persuasive":
            user_msg = (
                f"{self.persuasive_instruction}\n\n"
                f"{topic['prompt']}\n\n"
                f"The essay must be approximately {topic['words']} words.\n\n"
            )
        
        elif topic["type"]=="source-dependent":
            user_msg = (
                f"{self.source_instruction}\n\n"
                f"{topic['prompt']}\n\n"
                f"The essay must be approximately {topic['words']} words.\n\n"
            )
        prefill = "Essay:\n\n"

        tasks = [{
                "topic_id": topic["topic"],
                "type": "synthetic_generated",
                "system_msg": self.system_msg,
                "user_msg": user_msg,
                "prefill": prefill,
            } for _ in range(n_essays)]
        return tasks


class RewritePrompter:
    """Handles prompts for REWRITING existing essays."""
    def __init__(self):
        self.system_msg = "You are a strict, professional essay editor."
    
    def get_tasks(self, topic, n_essays):
        """Returns a list of task dicts for each existing human essay for the topic."""
        essays = topic.get('essays', [])
        essays=essays[:n_essays] #limiting to only the first n_essays
        tasks = []
        
        for i, essay in enumerate(essays):
            # Only rewrite human essays
            if essay.get('author') != 'human':
                continue 
            
            original_text = essay.get('text', "")
            if not isinstance(original_text, str) or len(original_text) < 10:
                continue 
                
            user_msg = (
                f"Polish and optimize the following essay:\n\n "
                f"{original_text}"
            )

            prefill = "Rewritten Essay:\n\n"
            
            tasks.append({
                "topic_id": topic["topic"],
                "type": "rewritten_essay",
                "system_msg": self.system_msg,
                "user_msg": user_msg,
                "prefill": prefill,
                "meta": {"original_author": essay.get('author'), "original_index": essay.get("essay_id")}
            })
        return tasks


# --- 2. MAIN SCRIPT ---

def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate or Rewrite essays based on Job Offers.")
    
    parser.add_argument("--mode", type=str, choices=["generate", "rewrite"], required=True,
                        help="Mode: 'generate' (new essays) or 'rewrite' (polish existing).")
    parser.add_argument("--model_name", type=str, default="Llama-3.1-8B-Instruct")
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, default="./Data/augmented_essays.json")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.9)
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    print("="*40)
    print(f"Mode:  {args.mode.upper()}")
    print(f"Model: {args.model_name}")
    print("="*40)

    # 1. Load Data
    topics = []
    try:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            topics = json.load(f)
    except Exception as e:
        print(f"Error loading JSON: {e}")
        sys.exit(1)
    
    print(f"Loaded {len(topics)} topics.")

    # 2. Load Model using govt_name mapping
    tokenizer = AutoTokenizer.from_pretrained(govt_name(args.model_name),)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    model = AutoModelForCausalLM.from_pretrained(
        govt_name(args.model_name),
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
        device_map="auto",
        attn_implementation="flash_attention_2" if torch.cuda.is_available() else "eager"
    )

    # 3. Prepare Tasks
    tasks = []
    if args.mode == "generate":
        prompter = GeneratePrompter()
        for topic in topics:
            # we generate 500 essays per topic
            tasks.extend(prompter.get_tasks(topic, n_essays=500))
            
    elif args.mode == "rewrite":
        prompter = RewritePrompter()
        for topic in topics:
            # We rewrite 500 essays per topic
            tasks.extend(prompter.get_tasks(topic, n_essays=500))

    print(f"Total tasks to process: {len(tasks)}")

    # 4. Construct Prompts
    print("Formatting prompts...")
    valid_tasks = []
    has_thinking_param = (tokenizer.chat_template is not None and "enable_thinking" in tokenizer.chat_template)
    
    for t in tqdm(tasks, desc="Formatting"):
        messages = [
            {"role": "system", "content": t.get("system_msg", "")},
            {"role": "user", "content": t.get("user_msg", "")}
        ]
        
        template_kwargs = {"conversation": messages, "tokenize": False, "add_generation_prompt": True}
        if has_thinking_param: template_kwargs["enable_thinking"] = False

        try:
            full_text = tokenizer.apply_chat_template(**template_kwargs)
            t["final_prompt_text"] = full_text + t["prefill"].lstrip()
            valid_tasks.append(t)
        except Exception as e:
            print(f"\n[Error] Could not format task for topic {t.get('topic', 'unknown')}: {e}")

    # 5. Generation Loop
    results_map = {} 

    print(f"\nStarting generation ({args.mode} mode)...")
    for i in tqdm(range(0, len(valid_tasks), args.batch_size)):
        batch = valid_tasks[i : i + args.batch_size]
        
        clean_prompts = [item["final_prompt_text"] for item in batch if "final_prompt_text" in item]
        if not clean_prompts: continue

        try:
            inputs = tokenizer(
                clean_prompts, return_tensors="pt", padding=True, truncation=True, max_length=16000
            ).to(model.device)
            
            input_len = inputs.input_ids.shape[1]

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=0.9,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            
            generated_ids = outputs[:, input_len:]
            decoded_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            
            for item, text in zip(batch, decoded_texts):
                tid = item["topic_id"]
                if tid not in results_map: results_map[tid] = []
                
                item["generated_text"] = item["prefill"] + text
                del item["final_prompt_text"] 
                del item["user_msg"]
                
                results_map[tid].append(item)
                
        except Exception as e:
            print(f" >> Batch Error: {e}")
            torch.cuda.empty_cache()

    # 6. Merge & Save
    print("Merging results...")
    final_output = []
    
    for topic in topics:
        tid=topic["topic"]
        generated_items = results_map.get(tid, [])
        
        if args.mode == "generate":
            prefix_len = len("Essay:\n\n")
            for i, item in enumerate(generated_items):
                final_text = item["generated_text"]
                if final_text.startswith("Essay:\n\n"):
                    final_text = final_text[prefix_len:]
                
                author_label = f"{args.model_name}_{args.temperature}"
                essay_id = f"{args.model_name}_{args.temperature}_{i}"
                topic['essays'].append({
                    "essay_id": essay_id,
                    "author": author_label,
                    "text": final_text.strip(),
                    #"persona_prompt": item["meta"]["persona"],
                    "score": None
                })

        elif args.mode == "rewrite":
            generated_items.sort(key=lambda x: x["meta"]["original_index"])
            prefix_len = len("Rewritten Essay:\n\n")
            
            for i, item in enumerate(generated_items):
                final_text = item["generated_text"]
                if final_text.startswith("Rewritten Essay:\n\n"):
                    final_text = final_text[prefix_len:]

                author_label = f"human_rewritten_{args.model_name}_{args.temperature}"
                essay_id = item["meta"]["original_index"] #putting back the original essay_id in case we want to compare
                topic['essays'].append({
                    "essay_id": essay_id,
                    "author": author_label,
                    "text": final_text.strip(),
                    "score":None
                })
        
        final_output.append(topic)

    print(f"Saving to {args.output_file}...")
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, indent=4, ensure_ascii=False)
    
    print("Done.")

if __name__ == "__main__":
    main()