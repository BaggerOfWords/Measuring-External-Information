import argparse
import json
import os
import re
from typing import Any, Dict, List, Optional

import torch
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoProcessor,
    AutoTokenizer,
)

# Optional fallback, matching your previous code style
try:
    from utils import govt_name
except ImportError:
    def govt_name(x):
        return x

BULLET_INSTRUCTION = """You are helping build a controlled benchmark for human-effort analysis.

Summarize the peer review below into exactly 8 concise bullet points, in this exact order:
1. Paper contribution
2. Main strengths
3. Main weaknesses
4. Evidence quality
5. Novelty
6. Clarity
7. Missing experiments
8. Final judgment

Requirements:
- Return exactly 8 bullet points.
- Keep each bullet concise and faithful to the review.
- Do not invent content not supported by the review.
- If a category is not clearly discussed in the review, say so briefly.
- Do not add any introduction or conclusion.
- Use the following exact format:

- Paper contribution: ...
- Main strengths: ...
- Main weaknesses: ...
- Evidence quality: ...
- Novelty: ...
- Clarity: ...
- Missing experiments: ...
- Final judgment: ...

Peer review:
\"\"\"{review_text}\"\"\"
"""

def clean_string(s: Any) -> str:
    if s is None:
        return ""
    s = str(s)
    s = s.replace("\x00", "")
    try:
        s.encode("utf-8")
    except UnicodeEncodeError:
        s = s.encode("utf-8", "ignore").decode("utf-8")
    return s.strip()

def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(obj: Any, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=4, ensure_ascii=False)

def normalize_summary_text(text: str) -> str:
    text = clean_string(text)
    text = re.sub(r"^```[a-zA-Z0-9]*\n?", "", text)
    text = re.sub(r"\n?```$", "", text)

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    cleaned_lines = []

    for line in lines:
        line = re.sub(r"^\s*\d+[\.\)]\s*", "", line)
        if not line.startswith("- "):
            line = "- " + line.lstrip("-*• ").strip()
        cleaned_lines.append(line)

    return "\n".join(cleaned_lines).strip()

def build_messages_text_only(review_text: str) -> List[Dict[str, str]]:
    user_prompt = BULLET_INSTRUCTION.format(review_text=review_text)
    return [
        {"role": "system", "content": "You are a precise scientific writing assistant. You follow formatting instructions exactly."},
        {"role": "user", "content": user_prompt},
    ]

def build_messages_multimodal(review_text: str) -> List[Dict[str, Any]]:
    user_prompt = BULLET_INSTRUCTION.format(review_text=review_text)
    return [
        {"role": "system", "content": [{"type": "text", "text": "You are a precise scientific writing assistant. You follow formatting instructions exactly."}]},
        {"role": "user", "content": [{"type": "text", "text": user_prompt}]},
    ]

def build_fallback_prompt(review_text: str) -> str:
    return "You are a precise scientific writing assistant.\n\n" + BULLET_INSTRUCTION.format(review_text=review_text)

def get_default_torch_dtype() -> torch.dtype:
    return torch.bfloat16 if torch.cuda.is_available() else torch.float32

class SummarizerBackend:
    def __init__(
        self,
        model_name: str,
        attn_implementation: Optional[str] = None,
        trust_remote_code: bool = True,
    ):
        self.model_name = govt_name(model_name)
        self.attn_implementation = attn_implementation
        self.trust_remote_code = trust_remote_code

        self.model = None
        self.tokenizer = None
        self.processor = None
        self.is_multimodal = False

        self._load()

    def _load(self) -> None:
        model_kwargs: Dict[str, Any] = {
            "torch_dtype": get_default_torch_dtype(),
            "trust_remote_code": self.trust_remote_code,
        }
        if torch.cuda.is_available():
            model_kwargs["device_map"] = "auto"
            if self.attn_implementation:
                model_kwargs["attn_implementation"] = self.attn_implementation

        print(f"Loading Model: {self.model_name}")

        # ATTEMPT 1: Pure Text Causal LM
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True, trust_remote_code=self.trust_remote_code)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **model_kwargs)
            self.is_multimodal = False
            print(" >> Successfully loaded as pure text CausalLM.")
            
        except Exception as e:
            print(f" >> CausalLM load failed ({e}). Falling back to multimodal VLM...")
            # ATTEMPT 2: Multimodal ImageTextToText
            self.processor = AutoProcessor.from_pretrained(self.model_name, trust_remote_code=self.trust_remote_code)
            self.tokenizer = getattr(self.processor, "tokenizer", self.processor)
            self.model = AutoModelForImageTextToText.from_pretrained(self.model_name, **model_kwargs)
            self.is_multimodal = True
            print(" >> Successfully loaded as multimodal ImageTextToText model.")

        if not torch.cuda.is_available():
            self.model.to("cpu")
        self.model.eval()

        # Safely handle Llama-3 style padding tokens
        if getattr(self.tokenizer, "pad_token_id", None) is None:
            if getattr(self.tokenizer, "eos_token_id", None) is not None:
                if isinstance(self.tokenizer.eos_token_id, list):
                    self.tokenizer.pad_token_id = self.tokenizer.eos_token_id[0]
                else:
                    self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def _model_device(self) -> torch.device:
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def generate_summary(
        self,
        review_text: str,
        max_new_tokens: int = 256,
        do_sample: bool = False,
        temperature: float = 0.7,
    ) -> str:
        
        # 1. Select the right prompt builder based on model architecture
        if self.is_multimodal:
            messages = build_messages_multimodal(review_text)
            template_engine = self.processor
        else:
            messages = build_messages_text_only(review_text)
            template_engine = self.tokenizer

        # 2. Apply Chat Template safely
        if getattr(template_engine, "chat_template", None):
            try:
                model_inputs = template_engine.apply_chat_template(
                    messages, tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True 
                )
            except TypeError:
                input_ids = template_engine.apply_chat_template(
                    messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
                )
                model_inputs = {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}
        else:
            prompt = build_fallback_prompt(review_text)
            toks = self.tokenizer(prompt, return_tensors="pt")
            model_inputs = dict(toks)

        # 3. Move inputs to GPU
        model_inputs = {k: v.to(self._model_device()) if hasattr(v, "to") else v for k, v in model_inputs.items()}
        prompt_len = model_inputs["input_ids"].shape[1]

        # 4. Generate kwargs
        gen_kwargs = {"max_new_tokens": max_new_tokens, "do_sample": do_sample}
        if do_sample:
            gen_kwargs["temperature"] = temperature
        if getattr(self.tokenizer, "pad_token_id", None) is not None:
            gen_kwargs["pad_token_id"] = self.tokenizer.pad_token_id

        # 5. Execute Generation
        with torch.no_grad():
            output_ids = self.model.generate(**model_inputs, **gen_kwargs)

        generated_only = output_ids[0, prompt_len:]
        decoded = self.tokenizer.decode(generated_only, skip_special_tokens=True)
        return normalize_summary_text(decoded)

def filter_human_reviews(reviews: List[Any]) -> List[Dict[str, Any]]:
    human_reviews = []
    ai_signatures = ["llama", "gpt", "qwen", "mistral", "gemini", "claude", "summary_", "rewritten_"]
    
    for review in reviews:
        if not isinstance(review, dict):
            human_reviews.append({"reviewer": "human", "text": str(review)})
            continue
            
        reviewer = clean_string(review.get("reviewer", "")).lower()
        
        if "human" in reviewer:
            human_reviews.append(review)
            continue
            
        if any(sig in reviewer for sig in ai_signatures):
            continue
            
        human_reviews.append(review)
        
    return human_reviews

def process_papers(
    papers: List[Dict[str, Any]],
    summarizer: SummarizerBackend,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
) -> List[Dict[str, Any]]:
    output_data = []

    for paper in tqdm(papers, desc="Summarizing human reviews"):
        paper_id = paper.get("id", "unknown_paper")
        paper_text = clean_string(paper.get("paper_text", ""))
        title = clean_string(paper.get("title", ""))

        human_reviews = filter_human_reviews(paper.get("reviews", []))
        if not human_reviews:
            continue

        summaries = []
        kept_reviews = []

        for review in human_reviews:
            original_review = clean_string(review.get("text", ""))
            if not original_review:
                continue

            kept_reviews.append(review)

            try:
                summary_text = summarizer.generate_summary(
                    review_text=original_review,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    temperature=temperature,
                )
                summaries.append({
                    "summary": summary_text,
                    "original review": original_review,
                })
            except Exception as e:
                summaries.append({
                    "summary": "",
                    "original review": original_review,
                    "error": str(e),
                })

        if not summaries:
            continue

        out_item = {
            "id": paper_id,
            "paper_text": paper_text,
            "reviews": kept_reviews,
            "summaries": summaries,
        }
        if title:
            out_item["title"] = title

        output_data.append(out_item)

    return output_data

def main():
    parser = argparse.ArgumentParser(description="Summarize human reviews into 8 structured bullet points.")
    parser.add_argument("--model_name", type=str, required=True, help="HF model name or local path.")
    parser.add_argument("--input_file", type=str, required=True, help="Input JSON file.")
    parser.add_argument("--output_file", type=str, required=True, help="Output JSON file.")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Maximum number of newly generated tokens per summary.")
    parser.add_argument("--do_sample", action="store_true", help="Use sampling instead of greedy decoding.")
    parser.add_argument("--temperature", type=float, default=0.9, help="Sampling temperature; used only with --do_sample.")
    parser.add_argument("--attn_implementation", type=str, default=None, help="Optional attention implementation, e.g. flash_attention_2.")
    parser.add_argument("--trust_remote_code", action="store_true", help="Enable trust_remote_code when loading processor/tokenizer/model.")

    args = parser.parse_args()

    print(f"Reading input file: {args.input_file}")
    data = load_json(args.input_file)
    if not isinstance(data, list):
        raise ValueError("Expected the input JSON to be a list of paper dictionaries.")

    summarizer = SummarizerBackend(
        model_name=args.model_name,
        attn_implementation=args.attn_implementation,
        trust_remote_code=args.trust_remote_code,
    )

    output_data = process_papers(
        papers=data,
        summarizer=summarizer,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
    )

    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    print(f"Saving {len(output_data)} papers to: {args.output_file}")
    save_json(output_data, args.output_file)
    print("Done.")

if __name__ == "__main__":
    main()