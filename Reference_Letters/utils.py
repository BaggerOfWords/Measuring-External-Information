import os

def govt_name(model_name):
    root = os.path.join(os.environ.get("DSDIR", ""), "HuggingFace_Models")
    patterns = [
        (lambda mn: "Tulu" in mn or "OLMo" in mn, "allenai/"),
        (lambda mn: "lama" in mn, "meta-llama/"),
        (lambda mn: "Hermes" in mn, "teknium/"),
        (lambda mn: "Mistral" in mn or "Mixtral" in mn, "mistralai/"),
        (lambda mn: "falcon" in mn or "Falcon" in mn, "tiiuae/"),
        (lambda mn: "bloom" in mn, "bigscience/"),
        (lambda mn: "galactica" in mn, "facebook/"),
        (lambda mn: "Capybara" in mn, "NousResearch/"),
        (lambda mn: "phi-2" in mn or "Orca-2" in mn or "Phi" in mn, "microsoft/"),
        (lambda mn: "miqu" in mn, "miqudev/"),
        (lambda mn: "Tower" in mn, "Unbabel/"),
        (lambda mn: "gpt-j" in mn or "gpt-neo" in mn, "EleutherAI/"),
        (lambda mn: "Qwen" in mn, "Qwen/"),
        (lambda mn: "gpt2" in mn, ""),
        (lambda mn: "gpt-oss" in mn, "openai/"),
        (lambda mn: "all-MiniLM" in mn, "sentence-transformers/"),
        (lambda mn: "gemma" in mn, "google/"),
        (lambda mn: "Qwopus" in mn, "Jackrong/")
    ]
    base_path = next((os.path.join(root, folder)
                      for condition, folder in patterns if condition(model_name)), "")
    return os.path.join(base_path, model_name) if base_path else model_name