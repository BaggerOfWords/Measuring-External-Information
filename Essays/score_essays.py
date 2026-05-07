import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForImageTextToText, AutoProcessor
import json
import argparse
import numpy as np
from tqdm import tqdm
import os
import random
import math
import re  
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.tokenize import sent_tokenize
import string

# Fallback for utils
try:
    from utils import govt_name
except ImportError:
    def govt_name(x): return x

# --- 1. DATA UTILS ---
def clean_string(s):
    if s is None: return ""
    s = str(s)
    if '\x00' in s: s = s.replace('\x00', '')
    try:
        s.encode('utf-8')
    except UnicodeEncodeError:
        s = s.encode('utf-8', 'ignore').decode('utf-8')
    return s

def flatten_data(data_items, target_arg=None): 
    flat_items = []
    
    # --- Resolve Target Author (Replacing Reviewer logic) ---
    target_author_name = None
    if target_arg is not None:
        try:
            target_idx = int(target_arg)
            is_index = True
        except ValueError:
            is_index = False
        
        if is_index:
            all_authors = set()
            for item in data_items:
                for essay in item.get('essays', []):
                    author = essay.get('author')
                    if author: all_authors.add(author)
            
            sorted_authors = sorted(list(all_authors))
            if 0 <= target_idx < len(sorted_authors):
                target_author_name = sorted_authors[target_idx]
                print(f" >> Index {target_idx} selected: '{target_author_name}'")
            else:
                print(f" !! Index {target_idx} out of bounds. Exiting.")
                return []
        else:
            target_author_name = target_arg
            print(f" >> Filtering for author: '{target_author_name}'")

    print("Flattening essay data...")
    
    for item in data_items:
        # Extract prompt and type
        prompt_text = clean_string(item.get('prompt', ''))
        essay_type = item.get('type', 'persuasive')
        
        # Build dynamic Context (C) based on type
        if essay_type == "persuasive":
            context_text = f"Act as a middle school student, please read the below prompt and write an essay: {prompt_text}"
        elif essay_type == "source-dependent":
            context_text = f"Act as a middle school student, please read the source essay and write an essay based on given topic prompt. {prompt_text}"
        else:
            # Fallback
            context_text = f"Act as a middle school student, please write an essay based on the following prompt: {prompt_text}"

        # Iterate through all essays for this topic/prompt
        essays = item.get('essays', [])
        for i, essay in enumerate(essays):
            author = essay.get('author', f"Unknown_Author_{i}")
            
            # Filter by author if one was specified
            if target_author_name is not None and author != target_author_name:
                continue

            target_text = clean_string(essay.get('text', ''))
            essay_id = clean_string(essay.get('essay_id', f"idx_{i}"))
            
            if len(target_text) < 50: continue 
            
            unique_id = f"{essay_id}_{author}"
            
            flat_items.append({
                "unique_id": unique_id,
                "author": author,
                "context_text": context_text,
                "target_text": target_text
            })
            
    print(f" >> Loaded {len(flat_items)} essays.")
    return flat_items

class EssayDataset(Dataset):
    def __init__(self, flat_data):
        self.data = flat_data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

def collate_fn(batch):
    return batch

# --- 2. SCORING ENGINE ---

class ContextualInfluenceScorer:
    def __init__(self, model, tokenizer, device, ratio, max_seq_len, hint_method, span_length, nb_repeats, max_context_tokens=-1, tfidf_vectorizer=None, num_sentences=3):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.ratio = ratio
        self.max_seq_len = max_seq_len
        self.hint_method = hint_method
        self.span_length = span_length
        self.nb_repeats = nb_repeats
        self.max_context_tokens = max_context_tokens
        self.tfidf_vectorizer = tfidf_vectorizer
        self.num_sentences = num_sentences
        
        # DEFINED VOCAB SIZE (Vital for safety check)
        self.vocab_size = self.model.config.vocab_size
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate_text_hints(self, text, method, ratio, span_length):
        words = text.split()
        n_words = len(words)
        if n_words == 0: return ""
        
        if method == "prefix":
            k = max(1, int(math.ceil(n_words * ratio)))
            return " ".join(words[:k])
            
        elif method == "suffix":
            k = max(1, int(math.ceil(n_words * ratio)))
            return " ".join(words[-k:])
            
        elif method == "random":
            k = max(1, int(math.ceil(n_words * ratio)))
            indices = sorted(random.sample(range(n_words), k))
            return " ".join([words[i] for i in indices])
            
        elif method == "random_spans":
            chunks = [words[i:i+span_length] for i in range(0, n_words, span_length)]
            k_chunks = max(1, int(math.ceil((n_words * ratio) / span_length)))
            k_chunks = min(k_chunks, len(chunks))
            indices = sorted(random.sample(range(len(chunks)), k_chunks))
            selected = []
            for i in indices:
                selected.extend(chunks[i])
            return " ".join(selected)
            
        elif method == "keywords":
            vec = self.tfidf_vectorizer.transform([text]).toarray()[0]
            feature_names = self.tfidf_vectorizer.get_feature_names_out()
            word_scores = {feature_names[i]: vec[i] for i in np.nonzero(vec)[0]}
            
            k = max(1, int(math.ceil(n_words * ratio)))
            scored_words = []
            
            for i, w in enumerate(words):
                clean_w = w.lower().strip(string.punctuation)
                score = word_scores.get(clean_w, 0.0)
                scored_words.append((score, i, w))
                
            scored_words.sort(key=lambda x: x[0], reverse=True)
            top_k = scored_words[:k]
            top_k.sort(key=lambda x: x[1])
            return " ".join([x[2] for x in top_k])
            
        elif method == "extractive":
            sentences = sent_tokenize(text)
            if len(sentences) <= self.num_sentences:
                return " ".join(sentences)
                
            vec = self.tfidf_vectorizer.transform([text]).toarray()[0]
            feature_names = self.tfidf_vectorizer.get_feature_names_out()
            word_scores = {feature_names[i]: vec[i] for i in np.nonzero(vec)[0]}
            
            sent_scores = []
            for i, sent in enumerate(sentences):
                s_words = [w.lower().strip(string.punctuation) for w in sent.split()]
                s_score = sum([word_scores.get(w, 0.0) for w in s_words]) / max(1, len(s_words))
                sent_scores.append((s_score, i, sent))
                
            sent_scores.sort(key=lambda x: x[0], reverse=True)
            top_n = sent_scores[:self.num_sentences]
            top_n.sort(key=lambda x: x[1])
            return " ".join([x[2] for x in top_n])
            
        return ""

    def generate_surprisal_hints(self, t_text, log_probs, ratio, span_length):
        tgt_tokens = self.tokenizer(f" {clean_string(t_text)}", add_special_tokens=False).input_ids
        
        # Safety check: if alignment fails for any reason, fallback to random_spans
        if len(tgt_tokens) != len(log_probs):
            print(f" >> [Warning] Tokenizer mismatch in surprisal logic. Falling back to random_spans.")
            return self.generate_text_hints(t_text, "random_spans", ratio, span_length)
            
        N = len(tgt_tokens)
        if N == 0: return ""
        
        k_tokens = max(1, int(math.ceil(N * ratio)))
        
        if span_length >= N:
            return self.tokenizer.decode(tgt_tokens)
            
        surprisal = -log_probs 
        
        window_scores = []
        for i in range(N - span_length + 1):
            window_scores.append((sum(surprisal[i:i+span_length]), i))
            
        window_scores.sort(key=lambda x: x[0], reverse=True)
        
        selected_mask = np.zeros(N, dtype=bool)
        selected_spans = []
        tokens_selected = 0
        
        for score, idx in window_scores:
            if tokens_selected >= k_tokens:
                break
            if not np.any(selected_mask[idx:idx+span_length]):
                selected_mask[idx:idx+span_length] = True
                selected_spans.append((idx, idx+span_length))
                tokens_selected += span_length
                
        selected_spans.sort(key=lambda x: x[0])
        
        hints = [self.tokenizer.decode(tgt_tokens[s:e]) for s, e in selected_spans]
        return " ... ".join(hints)

    def _get_token_logprobs(self, contexts, prompts, targets):
        # 1. Unwrap (Single Item)
        ctx, prm, tgt = contexts[0], prompts[0], targets[0]

        # 2. Tokenize directly to Tensor
        ctx_tokens = self.tokenizer(
            clean_string(ctx), 
            add_special_tokens=True, 
            truncation=True, 
            max_length=self.max_seq_len,
            return_tensors='pt'
        ).input_ids[0]

        # Bottleneck Truncation
        if self.max_context_tokens > 0 and len(ctx_tokens) > self.max_context_tokens:
            ctx_tokens = ctx_tokens[:self.max_context_tokens]

        # Adjust formatting (Changed 'Review:' to 'Essay:')
        prm_str = clean_string(prm)
        if prm_str:
            full_prm_str = f"\n\n{prm_str}\n\nEssay:"
        else:
            full_prm_str = "\n\nEssay:"

        prm_tokens = self.tokenizer(
            full_prm_str, 
            add_special_tokens=False, 
            return_tensors='pt'
        ).input_ids[0]

        tgt_tokens = self.tokenizer(
            f" {clean_string(tgt)}", 
            add_special_tokens=False, 
            return_tensors='pt'
        ).input_ids[0]

        # 3. Concatenate
        full_ids = torch.cat([ctx_tokens, prm_tokens, tgt_tokens])

        # Global Truncation Check
        if len(full_ids) > self.max_seq_len:
            keep_len = self.max_seq_len - len(prm_tokens) - len(tgt_tokens)
            if keep_len > 0:
                ctx_tokens = ctx_tokens[:keep_len]
                full_ids = torch.cat([ctx_tokens, prm_tokens, tgt_tokens])
            else:
                full_ids = full_ids[:self.max_seq_len]
                
        # 4. Build Labels
        ignore_prefix = torch.full((len(ctx_tokens) + len(prm_tokens),), -100, dtype=torch.long)
        full_labels = torch.cat([ignore_prefix, tgt_tokens])
        full_labels = full_labels[:len(full_ids)]

        # --- SAFETY CHECK (CPU Side) ---
        max_id = full_ids.max().item()
        if max_id >= self.vocab_size:
            print(f"\n[ERROR] Token ID {max_id} >= Vocab Size {self.vocab_size}")
            raise ValueError("Token ID Out of Bounds")
        # -------------------------------

        # 5. Move to GPU & Add Batch Dimension
        input_tensor = full_ids.unsqueeze(0).to(self.device)
        
        # 6. Forward Pass
        with torch.no_grad():
            outputs = self.model(input_tensor)
            logits = outputs.logits[0] 

        # 7. Calculate LogProbs
        shift_logits = logits[:-1, :]
        shift_labels = full_labels[1:].to(self.device)
        
        log_probs = F.log_softmax(shift_logits, dim=-1)

        # 8. Extract Target Probs
        safe_gather_indices = shift_labels.clone()
        safe_gather_indices[safe_gather_indices == -100] = 0
        
        target_log_probs = log_probs.gather(1, safe_gather_indices.unsqueeze(1)).squeeze(1)
        
        mask = (shift_labels != -100)
        final_scores = target_log_probs[mask]

        return [final_scores.cpu().float().numpy()], len(ctx_tokens)

    def score_batch(self, batch_data):
        # 1. Pull dynamic contexts and targets directly
        unified_contexts = [b['context_text'] for b in batch_data]
        target_texts = [b['target_text'] for b in batch_data]
        
        # 2. Baseline Run (No extra prompt)
        empty_prompts = [""] * len(batch_data)
        baseline_arrays, ctx_len = self._get_token_logprobs(unified_contexts, empty_prompts, target_texts)
        
        # 3. Handle Repeats and Hints Computation
        if self.hint_method in ["prefix", "suffix", "surprisal_spans", "keywords", "extractive"]:
            repeats = 1
        else:
            repeats = self.nb_repeats 
            
        expanded_contexts = []
        expanded_targets = []
        expanded_sparse_prompts = []
        
        for i, (ctx, t_text) in enumerate(zip(unified_contexts, target_texts)):
            if self.hint_method == "surprisal_spans":
                sparse = self.generate_surprisal_hints(t_text, baseline_arrays[i], self.ratio, self.span_length)
                for _ in range(repeats):
                    expanded_contexts.append(ctx)
                    expanded_targets.append(t_text)
                    expanded_sparse_prompts.append(f"Essay hints: {sparse}")
            else:
                for _ in range(repeats):
                    sparse = self.generate_text_hints(t_text, self.hint_method, self.ratio, self.span_length)
                    expanded_contexts.append(ctx)
                    expanded_targets.append(t_text)
                    expanded_sparse_prompts.append(f"Essay hints: {sparse}")
        
        # Process Expanded (1 by 1)
        all_sparse_arrays = []
        
        for k in range(len(expanded_contexts)):
            c_chunk = expanded_contexts[k : k+1]
            t_chunk = expanded_targets[k : k+1]
            s_chunk = expanded_sparse_prompts[k : k+1]
            
            chunk_results, _ = self._get_token_logprobs(c_chunk, s_chunk, t_chunk)
            all_sparse_arrays.extend(chunk_results)
            
        # 4. Deltas Calculation
        results_batch = []
        for i, item in enumerate(batch_data):
            base_arr = baseline_arrays[i]
            if len(base_arr) == 0: continue

            start = i * repeats
            end = start + repeats
            sparse_run_arrays = all_sparse_arrays[start:end]
            
            aligned_sparse_runs = []
            for run in sparse_run_arrays:
                if len(run) == 0: continue
                min_len = min(len(run), len(base_arr))
                aligned_sparse_runs.append(run[:min_len])
            
            if not aligned_sparse_runs: continue
                
            sparse_matrix = np.vstack(aligned_sparse_runs)
            avg_sparse_arr = np.mean(sparse_matrix, axis=0)
            
            final_len = avg_sparse_arr.shape[0]
            truncated_base = base_arr[:final_len]
            
            res_item = {
                "id": item['unique_id'],    
                "author": item['author'], 
                "context_length": ctx_len,
                "baseline_logprobs": np.round(truncated_base, 6).tolist(),
                "hinted_logprobs": np.round(avg_sparse_arr, 6).tolist()
            }
            results_batch.append(res_item)
            
        return results_batch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Llama-3.1-8B-Instruct") 
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, default="influence_scores.json")
    parser.add_argument("--ratio", type=float, default=0.2)
    parser.add_argument("--hint_method", type=str, default="random", choices=["random", "random_spans", "prefix", "suffix", "surprisal_spans", "keywords", "extractive"], help="Method for selecting hints.")
    parser.add_argument("--num_sentences", type=int, default=3, help="Number of sentences for extractive summaries.")
    parser.add_argument("--span_length", type=int, default=5)
    parser.add_argument("--nb_repeats", type=int, default=10)
    parser.add_argument("--max_seq_len", type=int, default=8192) 
    parser.add_argument("--max_context_tokens", type=int, default=-1, help="Truncate context (prompt+source) to this many tokens.")
    parser.add_argument("--target_author", type=str, default=None, help="Filter by author string or zero-indexed integer")
    args = parser.parse_args()

    print(f"--- Scoring Essays (Token-Level Deltas) ---")
    print(f"Input: {args.input_file}")
    print(f"Hint Method: {args.hint_method} (Ratio: {args.ratio}, Span Length: {args.span_length}, Repeats: {args.nb_repeats})")

    with open(args.input_file, 'r') as f:
        raw_data = json.load(f)
    
    # Check if data is just a dict, wrap in list if so
    if isinstance(raw_data, dict):
        raw_data = [raw_data]
        
    flat_data = flatten_data(raw_data, target_arg=args.target_author)

    tfidf_vectorizer = None
    if args.hint_method in ["keywords", "extractive"]:
        print("Fitting TF-IDF on the essay corpus...")
        corpus = [item["target_text"] for item in flat_data]
        tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_vectorizer.fit(corpus)
    
    try:
        model_path = govt_name(args.model_name)
    except:
        model_path = args.model_name
        
    print(f"Loading Model: {model_path}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    except Exception:
        print(" >> AutoTokenizer failed. Loading via AutoProcessor...")
        processor = AutoProcessor.from_pretrained(model_path)
        tokenizer = getattr(processor, "tokenizer", processor)
        
    model_kwargs = {
        "device_map": "auto",
        "torch_dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float16,
        "attn_implementation": "flash_attention_2" if torch.cuda.is_available() else "eager"
    }

    try:
        model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
    except ValueError:
        print(" >> AutoModelForCausalLM failed. Falling back to AutoModelForImageTextToText...")
        model = AutoModelForImageTextToText.from_pretrained(model_path, **model_kwargs)
    model.eval()

    scorer = ContextualInfluenceScorer(
        model, tokenizer, model.device, 
        ratio=args.ratio, 
        max_seq_len=args.max_seq_len,
        hint_method=args.hint_method,
        span_length=args.span_length,
        nb_repeats=args.nb_repeats,
        max_context_tokens=args.max_context_tokens,
        tfidf_vectorizer=tfidf_vectorizer,
        num_sentences=args.num_sentences
    )
    
    dataset = EssayDataset(flat_data)
    loader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn)

    final_results = []
    
    for batch in tqdm(loader, desc="Scoring"):
        try:
            scores = scorer.score_batch(batch)
            final_results.extend(scores)
        except Exception as e:
            print(f" >> Critical Error skipping batch: {e}")

    print(f"Saving {len(final_results)} items to {args.output_file}...")
    with open(args.output_file, 'w') as f:
        json.dump(final_results, f, indent=4)
    print("Done.")

if __name__ == "__main__":
    main()