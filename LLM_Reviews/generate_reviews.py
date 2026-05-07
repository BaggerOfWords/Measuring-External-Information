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

class ReviewPrompter:
    """Handles prompts for generating NEW reviews from paper text."""
    def __init__(self):
        self.base_instruction = (
            "You are an expert reviewer for the ICLR 2019 Conference. "
            "Your task is to write a high-quality peer review. "
            "Critique based on: Clarity, Quality, Novelty, and Significance."
        )
        self.personas = [
            "Provide a balanced review. List strengths and weaknesses explicitly.",
            "Focus heavily on the experiments and empirical results. Be skeptical.",
            "Focus primarily on novelty and theoretical contribution.",
            "Pay special attention to clarity and structure.",
            "Be a constructive mentor. Offer specific fixes."
        ]

    def get_tasks(self, paper, n_reviews):
        """Returns a list of task dicts for this paper."""
        paper_text = paper.get('text', "")
        if paper_text is None: paper_text = ""
        
        tasks = []
        selected_personas = []
        while len(selected_personas) < n_reviews:
            chunk = self.personas[:]
            random.shuffle(chunk)
            selected_personas.extend(chunk)
        selected_personas = selected_personas[:n_reviews]

        for i, persona in enumerate(selected_personas):
            system_msg = "You are a reviewer for ICLR 2019."
            user_msg = (
                f"{self.base_instruction}\n\n"
                f"REVIEWER FOCUS: {persona}\n\n"
                f"--- BEGIN PAPER CONTENT ---\n"
                f"{paper_text[:50000]}" 
                f"\n--- END PAPER CONTENT ---\n\n"
                "Write your official ICLR 2019 review now."
            )
            prefill = "Review for ICLR 2019:\n\nSummary:\n"
            
            tasks.append({
                "paper_id": paper['id'],
                "type": "synthetic_review",
                "system_msg": system_msg,
                "user_msg": user_msg,
                "prefill": prefill,
                "meta": {"persona": persona, "index": i}
            })
        return tasks


class RewritePrompter:
    """Handles prompts for REWRITING existing reviews for clarity."""
    def __init__(self):
        self.system_msg = "You are a professional scientific editor."
    
    def get_tasks(self, paper):
        """Returns a list of task dicts for each existing review in the paper."""
        reviews = paper.get('reviews', [])
        tasks = []
        
        for i, review in enumerate(reviews):
            # --- MODIFICATION 1: Filter for Human Reviews Only ---
            if review.get('reviewer') != 'human':
                continue 
            
            original_text = review.get('text', "")
            if not isinstance(original_text, str) or len(original_text) < 10:
                continue # Skip empty/invalid reviews
                
            user_msg = (
                "Your task is to rewrite the following peer review to improve its clarity, flow, and grammar. "
                "Do NOT change the technical content, the sentiment (positive/negative), or the scores. "
                "Keep the critique exactly as it is, just make it easier to read.\n\n"
                f"--- ORIGINAL REVIEW ---\n{original_text}\n--- END REVIEW ---\n\n"
                "Provide the rewritten version below."
            )
            
            # Prefill forces the model to skip "Sure, here is the text"
            prefill = "Rewritten Review:\n\n"
            
            tasks.append({
                "paper_id": paper['id'],
                "type": "rewritten_review",
                "system_msg": self.system_msg,
                "user_msg": user_msg,
                "prefill": prefill,
                "meta": {"original_reviewer": review.get('reviewer'), "original_index": i}
            })
        return tasks
    
class CompletionPrompter:
    """Handles prompts for COMPLETING reviews cut off at k% tokens."""
    def __init__(self, tokenizer, k_percent):
        self.tokenizer = tokenizer
        self.k_percent = k_percent
        self.system_msg = "You are an expert ICLR reviewer."

    def get_tasks(self, paper):
        reviews = paper.get('reviews', [])
        tasks = []
        paper_text = paper.get('text', "")[:50000] # Truncate paper context if too long

        for i, review in enumerate(reviews):
            if review.get('reviewer') != 'human':
                continue
            original_text = review.get('text', "")
            if not isinstance(original_text, str) or len(original_text) < 10:
                continue

            # 1. Tokenize the original review
            tokens = self.tokenizer.encode(original_text, add_special_tokens=False)
            
            # 2. Calculate cut-off
            cut_index = int(len(tokens) * (self.k_percent / 100.0))
            if cut_index == 0: cut_index = 1 # Keep at least 1 token
            
            # 3. Decode back to text to create the "Prefill"
            # This ensures the model sees exactly the tokens intended
            partial_text = self.tokenizer.decode(tokens[:cut_index])

            user_msg = (
                "Below is the text of a submission to ICLR 2019.\n"
                f"--- BEGIN PAPER CONTENT ---\n{paper_text}\n--- END PAPER CONTENT ---\n\n"
                "A reviewer started writing a review for this paper but stopped halfway. "
                "Please complete the review naturally, maintaining the style, tone, and logical flow of the existing text.\n"
                "Do NOT repeat the start of the review. Start exactly where the text cuts off below."
            )

            tasks.append({
                "paper_id": paper['id'],
                "type": "completion_review",
                "system_msg": self.system_msg,
                "user_msg": user_msg,
                "prefill": partial_text, # The model will force-start with this
                "meta": {
                    "original_reviewer": review.get('reviewer'), 
                    "original_index": i,
                    "cut_percentage": self.k_percent
                }
            })
        return tasks

# --- 2. MAIN SCRIPT ---

def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate, Rewrite, or Complete ICLR Reviews.")
    
    # Updated choices to include 'completion'
    parser.add_argument("--mode", type=str, choices=["review", "rewrite", "completion"], required=True,
                        help="Mode: 'review' (new), 'rewrite' (polish), or 'completion' (finish partial).")
    
    parser.add_argument("--completion_k", type=int, default=50, 
                        help="For completion mode: Percentage of tokens to keep (0-100). Default 50.")
    
    parser.add_argument("--model_name", type=str, default="Llama-3.1-8B-Instruct")
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, default="results.json")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.7)
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    print("="*40)
    print(f"Mode:  {args.mode.upper()}")
    print(f"Model: {args.model_name}")
    print("="*40)

    # 1. Load Data
    papers = []
    try:
        with open(args.input_file, 'r') as f:
            # Check file extension to decide how to load
            if args.input_file.endswith('.jsonl'):
                print("Detected JSONL format.")
                for line in f:
                    if line.strip(): # Skip empty lines
                        papers.append(json.loads(line))
            else:
                # Assume standard JSON list (starts with [)
                print("Detected standard JSON format.")
                papers = json.load(f)
                
    except FileNotFoundError:
        print(f"Error: Could not find {args.input_file}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        print("Tip: Check if your file is .json (list) or .jsonl (lines).")
        sys.exit(1)
    
    print(f"Loaded {len(papers)} papers.")

    # 2. Load Model
    # Use use_fast=False if you encounter tokenization errors on your cluster
    tokenizer = AutoTokenizer.from_pretrained(govt_name(args.model_name))
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    model = AutoModelForCausalLM.from_pretrained(
        govt_name(args.model_name),
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
        device_map="auto",
        attn_implementation="flash_attention_2" if torch.cuda.is_available() else "eager"
    )
    # 3. Prepare Tasks based on Mode
    tasks = []
    
    if args.mode == "review":   
        prompter = ReviewPrompter()
        for paper in papers:
            real_reviews = paper.get('reviews', [])
            
            # Count exactly how many human reviews this specific paper has
            human_count = sum(1 for r in real_reviews if isinstance(r, dict) and r.get('reviewer') == 'human')
            
            # Only generate tasks if there is actually a human review to match
            if human_count > 0:
                tasks.extend(prompter.get_tasks(paper, human_count))
            
    elif args.mode == "rewrite":
        prompter = RewritePrompter()
        for paper in papers:
            tasks.extend(prompter.get_tasks(paper))

    elif args.mode == "completion":
        prompter = CompletionPrompter(tokenizer, args.completion_k)
        for paper in papers:
            tasks.extend(prompter.get_tasks(paper))

    print(f"Total items to process: {len(tasks)}")



    # 4. Construct Prompts (Standardized)
    print("Formatting prompts...")
    valid_tasks = []

    has_thinking_param = (
        tokenizer.chat_template is not None 
        and "enable_thinking" in tokenizer.chat_template
    )
    
    for t in tqdm(tasks, desc="Formatting"):
        # 2. Data Validation: Ensure roles are correct
        messages = [
            {"role": "system", "content": t.get("system_msg", "")},
            {"role": "user", "content": t.get("user_msg", "")}
        ]
        
        # 3. Dynamic Argument Construction
        template_kwargs = {
            "conversation": messages,
            "tokenize": False, 
            "add_generation_prompt": True
        }
        
        if has_thinking_param:
            template_kwargs["enable_thinking"] = False

        # 4. Pure Template Execution
        try:
            full_text = tokenizer.apply_chat_template(**template_kwargs)
            
            # Prefill Logic: Connect the template to your custom start string
            t["final_prompt_text"] = full_text + t["prefill"].lstrip()
            valid_tasks.append(t)
            
        except Exception as e:
            # If the model does not use the 'system' role (like Gemma), merge it into 'user' and retry
            if "system" in str(e).lower():
                merged_content = f"{t.get('system_msg', '')}\n\n{t.get('user_msg', '')}"
                template_kwargs["conversation"] = [{"role": "user", "content": merged_content.strip()}]
                
                try:
                    full_text = tokenizer.apply_chat_template(**template_kwargs)
                    t["final_prompt_text"] = full_text + t["prefill"].lstrip()
                    valid_tasks.append(t)
                except Exception as fallback_e:
                    print(f"\n[Error] Fallback failed for paper {t.get('paper_id', 'unknown')}: {fallback_e}")
            else:
                print(f"\n[Error] Could not format task for paper {t.get('paper_id', 'unknown')}: {e}")

    # 5. Generation Loop
    results_map = {} # Key: paper_id, Value: List of results

    print(f"\nStarting generation ({args.mode} mode)...")
    
    for i in tqdm(range(0, len(valid_tasks), args.batch_size)):
        batch = valid_tasks[i : i + args.batch_size]
        
        # Filter bad prompts
        clean_prompts = []
        clean_batch_items = []
        for item in batch:
            p_text = item.get("final_prompt_text")
            if isinstance(p_text, str) and len(p_text) > 0:
                clean_prompts.append(p_text)
                clean_batch_items.append(item)

        if not clean_prompts: continue

        try:
            inputs = tokenizer(
                clean_prompts, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=32000
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
            
            for item, text in zip(clean_batch_items, decoded_texts):
                pid = item["paper_id"]
                if pid not in results_map: results_map[pid] = []
                
                # Store result
                item["generated_text"] = item["prefill"] + text
                item["machine_part"] = text
                
                # Cleanup huge prompt to save RAM before saving
                del item["final_prompt_text"] 
                del item["user_msg"]
                
                results_map[pid].append(item)
                
        except Exception as e:
            print(f" >> Batch Error: {e}")
            torch.cuda.empty_cache()

    # 6. Merge & Save
    print("Merging results...")
    final_output = []
    
    for paper in papers:
        pid = paper['id']
        generated_items = results_map.get(pid, [])
        
        if args.mode == "review":
            # 1. Define the prefill length to strip it out
            prefix_len = len("Review for ICLR 2019:\n\nSummary:\n")
            
            for item in generated_items:
                # 2. Clean the text
                final_text = item["generated_text"]
                if final_text.startswith("Review for ICLR 2019:\n\nSummary:\n"):
                    final_text = final_text[prefix_len:]
                
                # 3. Create a standardized reviewer label
                persona_idx = item["meta"]["index"]
                reviewer_label = f"synthetic_{args.model_name}_temp_{args.temperature}"

                # 4. Create the standardized entry
                new_review_entry = {
                    "reviewer": reviewer_label,
                    "text": final_text.strip(),
                    "persona_prompt": item["meta"]["persona"]
                }
                
                # 5. Append to the main reviews list 
                paper['reviews'].append(new_review_entry)
        elif args.mode == "rewrite":
            # Sort by original index to keep relative order
            generated_items.sort(key=lambda x: x["meta"]["original_index"])
            
            # Save with specific signer ---
            reviewer_label = f"human_rewritten_{args.model_name}_temp_{args.temperature}"
            
            # Remove the "Rewritten Review:\n\n" prefix from the final text 
            # so it looks like a clean review in the dataset
            prefix_len = len("Rewritten Review:\n\n")

            for item in generated_items:
                # Clean the text (remove the prefill header we forced the model to generate)
                final_text = item["generated_text"]
                if final_text.startswith("Rewritten Review:\n\n"):
                    final_text = final_text[prefix_len:]

                new_review_entry = {
                    "reviewer": reviewer_label,
                    "text": final_text
                }
                
                # Append to the MAIN reviews list
                paper['reviews'].append(new_review_entry)
        
        elif args.mode == "completion":
            generated_items.sort(key=lambda x: x["meta"]["original_index"])
            paper['completed_reviews'] = [
                {
                    "reviewer": f"human_completed_{args.model_name}_{item['meta']['cut_percentage']}percent_temp_{args.temperature}",
                    "text": item["generated_text"],
                    "human part": item["prefill"],
                    "machine part": item["machine_part"]
                }
                for item in generated_items
            ]
            
        final_output.append(paper)

    print(f"Saving to {args.output_file}...")
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, indent=4)
    
    print("Done.")

if __name__ == "__main__":
    main()