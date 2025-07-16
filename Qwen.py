# %%
# Plotly needs a different renderer for VSCode/Notebooks vs Colab argh
import plotly.io as pio
pio.renderers.default = "notebook_connected"
print(f"Using renderer: {pio.renderers.default}")


import torch
torch.set_grad_enabled(False)

from transformers import AutoTokenizer
from transformer_lens import HookedTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig

from functools import partial

# %%
def assert_hf_and_tl_model_are_close(
    hf_model,
    tl_model,
    tokenizer,
    prompt="This is a prompt to test out",
    atol=1e-3,
):
    prompt_toks = tokenizer(prompt, return_tensors="pt").input_ids

    hf_logits = hf_model(prompt_toks.to(hf_model.device)).logits
    tl_logits = tl_model(prompt_toks).to(hf_logits)

    assert torch.allclose(torch.softmax(hf_logits, dim=-1), torch.softmax(tl_logits, dim=-1), atol=atol)

# %% [markdown]
# ## Qwen, first generation

# %%
model_path = "Qwen/Qwen-1_8B-Chat"
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    trust_remote_code=True
)

hf_model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map=device,
    fp32=True,
    use_logn_attn=False,
    use_dynamic_ntk = False,
    scale_attn_weights = False,
    trust_remote_code = True
).eval()

tl_model = HookedTransformer.from_pretrained_no_processing(
    model_path,
    device=device,
    fp32=True,
    dtype=torch.float32,
).to(device)

assert_hf_and_tl_model_are_close(hf_model, tl_model, tokenizer)

# %% [markdown]
# ## Qwen, new generation

# %%
model_path = "Qwen/Qwen1.5-1.8B-Chat"
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(
    model_path,
)

hf_model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map=device,
).eval()

tl_model = HookedTransformer.from_pretrained_no_processing(
    model_path,
    device=device,
    dtype=torch.float32,
).to(device)

assert_hf_and_tl_model_are_close(hf_model, tl_model, tokenizer)

# %%



