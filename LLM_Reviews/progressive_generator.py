import argparse
import json
import os
import re
from typing import Any, Dict, List

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForImageTextToText, AutoProcessor

# Optional fallback
try:
    from utils import govt_name
except ImportError:
    def govt_name(x): return x

# --- PROMPTS ---
REVIEWER_GUIDELINE = """
Publication standards. It’s important to maintain the high quality of ICLR papers. At the same time, we must be cognizant of the fact that different reviewers, authors, and even different areas will have different standards about what constitutes a high-quality paper. Reviewers are encouraged to use their best judgement, but also ask themselves the following questions:
    - Does the paper present substantively new ideas or explore an underexplored or highly novel question? 
    - Will a substantial fraction of the attendees be interested in reading this paper?
    - Would I send this paper to one of my colleagues to read? 

At the same time, it is critical to maintain a high standard in terms of scientific rigor: if you believe that a paper has flaws in terms of its evaluation or validation, proofs, or other parts of the discussion, it is critical to point this out to the authors.
"""

SYSTEM_PROMPT = f"""You are an AI researcher reviewing a paper submitted to a prestigious AI research conference.
You will be provided with the manuscript text, the conference's reviewer guidelines, and a partial list of core arguments/bullet points extracted from a previous human review.

Your objective is to thoroughly evaluate the paper, adhering to the provided guidelines.
CRITICAL INSTRUCTIONS:
1. You must use the provided bullet points as the absolute foundation of your review. 
2. Expand upon these specific points using the manuscript text to ground your arguments in reality.
3. Your final assessment MUST perfectly reflect the exact sentiments of the provided bullet points. Do not invent major critiques or praises that contradict them.
4. Even though you may only receive a few bullet points, you must write a complete, cohesive peer review.
5. DO NOT output any internal thinking process, reasoning, or preamble. Output ONLY the final review text.

Ensure your evaluation is objective, comprehensive, and aligned with the conference standards.
{REVIEWER_GUIDELINE}"""

USER_PROMPT = """Target Review Length: Approximately {target_word_count} words.

Here are the core arguments you MUST include and expand upon in your review:
---
{partial_bullets}
---

Here is the paper you are asked to review. Write a well-justified review of this paper that incorporates the core arguments above. 
OUTPUT ONLY THE REVIEW TEXT. NO PREAMBLE. NO THINKING PROCESS.
---
{text}
---"""

# --- UTILITIES ---
def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(obj: Any, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=4, ensure_ascii=False)

def extract_bullets(summary_text: str) -> List[str]:
    """STRICT bullet extraction to prevent grabbing intro sentences."""
    lines = [line.strip() for line in summary_text.splitlines() if line.strip()]
    bullets = []
    for line in lines:
        # Strictly only accept lines that genuinely start as bullets
        if line.startswith("- ") or line.startswith("* "):
            bullets.append(line)
    return bullets[:8]

def get_word_count(text: str) -> int:
    return len(str(text).split())

def clean_generation(text: str) -> str:
    """Strips out Qwen/DeepSeek thinking blocks and preambles."""
    # 1. Strip raw <think> tags if the model uses them
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    
    # 2. Strip explicit "Thinking Process" blocks (splits at the word "Review:" if it exists)
    if re.search(r"Thinking Process:.*?Review:", text, flags=re.DOTALL | re.IGNORECASE):
        text = re.split(r"Review:", text, maxsplit=1, flags=re.IGNORECASE)[-1]
    else:
        # If no "Review:" marker, just try to strip the thinking block
        text = re.sub(r"^.*?Thinking Process:.*?(?=\n\n|\n#|\n\*\*)", "", text, flags=re.DOTALL | re.IGNORECASE)
        
    return text.strip()

# --- GENERATOR CLASS ---
class ProgressiveReviewGenerator:
    def __init__(self, model_name: str, attn_implementation: str = None):
        self.model_name = govt_name(model_name)
        
        print(f"Loading tokenizer: {self.model_name}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, 
                use_fast=True,
                trust_remote_code=True
            )
        except Exception:
            print(" >> AutoTokenizer failed. Loading via AutoProcessor...")
            self.processor = AutoProcessor.from_pretrained(
                self.model_name, 
                trust_remote_code=True
            )
            # Extract the text tokenizer from the multimodal processor
            self.tokenizer = getattr(self.processor, "tokenizer", self.processor)
        
        self.tokenizer.padding_side = "left"
        
        self.stop_token_ids = []
        if getattr(self.tokenizer, "eos_token_id", None) is not None:
            if isinstance(self.tokenizer.eos_token_id, list):
                self.stop_token_ids.extend(self.tokenizer.eos_token_id)
            else:
                self.stop_token_ids.append(self.tokenizer.eos_token_id)
                
        im_end_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        if im_end_id is not None and im_end_id not in self.stop_token_ids:
            self.stop_token_ids.append(im_end_id)

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.stop_token_ids[0] if self.stop_token_ids else 0

        kwargs = {
            "torch_dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            "trust_remote_code": True 
        }
        
        if torch.cuda.is_available():
            kwargs["device_map"] = "balanced" 
            if attn_implementation:
                kwargs["attn_implementation"] = attn_implementation

        print(f"Loading LM: {self.model_name}")
        try:
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **kwargs)
        except ValueError:
            print("AutoModelForCausalLM failed. Falling back to AutoModelForImageTextToText...")
            self.model = AutoModelForImageTextToText.from_pretrained(self.model_name, **kwargs)
            
        self.model.eval()

    def generate_batch(self, paper_text: str, list_partial_bullets: List[str], target_words: int) -> List[str]:
        safe_paper_text = paper_text[:80000] 
        
        batch_strings = []
        for bullets in list_partial_bullets:
            user_content = USER_PROMPT.format(
                target_word_count=target_words,
                partial_bullets=bullets,
                text=safe_paper_text
            )
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content}
            ]
            
            try:
                # Try standard text format (works for Llama-3, Qwen, etc.)
                prompt_str = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
                )
            except Exception:
                # Fallback for models (like Mistral 3.2 / Gemma 4) enforcing multimodal list formats
                messages_multi = [
                    {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                    {"role": "user", "content": [{"type": "text", "text": user_content}]}
                ]
                prompt_str = self.tokenizer.apply_chat_template(
                    messages_multi, tokenize=False, add_generation_prompt=True, enable_thinking=False
                )
                
            batch_strings.append(prompt_str)

        model_inputs = self.tokenizer(batch_strings, return_tensors="pt", padding=True).to(self.model.device)
        prompt_len = model_inputs["input_ids"].shape[1]

        max_tokens = min(4096, int(target_words * 2.0) + 200)

        with torch.no_grad():
            output_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.stop_token_ids 
            )

        generated_only = output_ids[:, prompt_len:]
        results = self.tokenizer.batch_decode(generated_only, skip_special_tokens=True)
        
        # Clean the thinking processes before returning
        return [clean_generation(res) for res in results]

# --- MAIN PROCESS ---
def process_progressive_reviews(input_file: str, output_file: str, generator: ProgressiveReviewGenerator, batch_size: int):
    if os.path.exists(output_file):
        print(f"Found existing progress in {output_file}. Resuming from save state...")
        try:
            data = load_json(output_file)
        except json.JSONDecodeError:
            print("Warning: Output file was corrupted. Starting fresh from input file.")
            data = load_json(input_file)
    else:
        print(f"Loading fresh data from {input_file}...")
        data = load_json(input_file)
    
    total_tasks = 0
    for paper in data:
        for summary_obj in paper.get("summaries", []):
            if summary_obj.get("summary"):
                for k in range(1, 9):
                    if f"generated_review_{k}_bullets" not in summary_obj or len(str(summary_obj.get(f"generated_review_{k}_bullets", ""))) < 50:
                        total_tasks += 1

    if total_tasks == 0:
        print("All progressive reviews are already generated!")
        return

    print(f"Starting batched generation for {total_tasks} missing reviews...")
    
    with tqdm(total=total_tasks, desc="Generating Reviews") as pbar:
        for paper in data:
            paper_text = paper.get("paper_text", "")
            
            for summary_obj in paper.get("summaries", []):
                summary_text = summary_obj.get("summary", "")
                if not summary_text:
                    continue
                
                original_review = summary_obj.get("original review", "")
                target_word_count = get_word_count(original_review)
                
                bullets = extract_bullets(summary_text)
                if not bullets:
                    # Skip if no valid bullets found after strict extraction
                    continue
                
                max_k = min(8, len(bullets))
                pending_ks = []
                pending_bullets = []
                
                for k in range(1, max_k + 1):
                    key_name = f"generated_review_{k}_bullets"
                    if key_name in summary_obj and len(str(summary_obj.get(key_name, ""))) > 50:
                        continue
                        
                    pending_ks.append(k)
                    pending_bullets.append("\n".join(bullets[:k]))

                if not pending_ks:
                    continue

                for i in range(0, len(pending_ks), batch_size):
                    ks_chunk = pending_ks[i:i+batch_size]
                    bullets_chunk = pending_bullets[i:i+batch_size]
                    
                    try:
                        generated_reviews = generator.generate_batch(
                            paper_text=paper_text,
                            list_partial_bullets=bullets_chunk,
                            target_words=target_word_count
                        )
                        for k, rev in zip(ks_chunk, generated_reviews):
                            summary_obj[f"generated_review_{k}_bullets"] = rev
                    except Exception as e:
                        print(f"\n[Error] Failed on paper {paper.get('id')} for chunk {ks_chunk}: {e}")
                        for k in ks_chunk:
                            summary_obj[f"generated_review_{k}_bullets"] = ""
                    
                    save_json(data, output_file)
                    pbar.update(len(ks_chunk))

    print(f"Finished! Saved batched progressive reviews to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--attn_implementation", type=str, default="flash_attention_2")
    parser.add_argument("--batch_size", type=int, default=4, help="Number of reviews to generate simultaneously")
    args = parser.parse_args()

    generator = ProgressiveReviewGenerator(
        model_name=args.model_name,
        attn_implementation=args.attn_implementation
    )
    
    process_progressive_reviews(args.input_file, args.output_file, generator, args.batch_size)