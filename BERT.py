# %%
# Plotly needs a different renderer for VSCode/Notebooks vs Colab argh
import plotly.io as pio
pio.renderers.default = "notebook_connected"
print(f"Using renderer: {pio.renderers.default}")

# %%
import circuitsvis as cv

# Testing that the library works
cv.examples.hello("Neel")

# %%
# Import stuff
import torch

from transformers import AutoTokenizer

from transformer_lens import HookedEncoder, BertNextSentencePrediction

# %%
torch.set_grad_enabled(False)

# %% [markdown]
# # BERT
# 
# In this section, we will load a pretrained BERT model and use it for the Masked Language Modelling and Next Sentence Prediction task

# %%
# NBVAL_IGNORE_OUTPUT
bert = HookedEncoder.from_pretrained("bert-base-cased")
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

# %% [markdown]
# ## Masked Language Modelling
# Use the "[MASK]" token to mask any tokens which you would like the model to predict.  
# When specifying return_type="predictions" the prediction of the model is returned, alternatively (and by default) the function returns logits.  
# You can also specify None as return type for which nothing is returned

# %%
prompt = "The [MASK] is bright today."

prediction = bert(prompt, return_type="predictions")

print(f"Prompt: {prompt}")
print(f'Prediction: "{prediction}"')

# %% [markdown]
# You can also input a list of prompts:

# %%
prompts = ["The [MASK] is bright today.", "She [MASK] to the store.", "The dog [MASK] the ball."]

predictions = bert(prompts, return_type="predictions")

print(f"Prompt: {prompts}")
print(f'Prediction: "{predictions}"')

# %% [markdown]
# ## Next Sentence Prediction
# To carry out Next Sentence Prediction, you have to use the class BertNextSentencePrediction, and pass a HookedEncoder in its constructor.  
# Then, create a list with the two sentences you want to perform NSP on as elements and use that as input to the forward function.  
# The model will then predict the probability of the sentence at position 1 following (i.e. being the next sentence) to the sentence at position 0.

# %%
nsp = BertNextSentencePrediction(bert)
sentence_a = "A man walked into a grocery store."
sentence_b = "He bought an apple."

input = [sentence_a, sentence_b]

predictions = nsp(input, return_type="predictions")

print(f"Sentence A: {sentence_a}")
print(f"Sentence B: {sentence_b}")
print(f'Prediction: "{predictions}"')

# %% [markdown]
# # Inputting tokens directly
# You can also input tokens instead of a string or a list of strings into the model, which could look something like this

# %%
prompt = "The [MASK] is bright today."

tokens = tokenizer(prompt, return_tensors="pt")["input_ids"]
logits = bert(tokens) # Since we are not specifying return_type, we get the logits
logprobs = logits[tokens == tokenizer.mask_token_id].log_softmax(dim=-1)
prediction = tokenizer.decode(logprobs.argmax(dim=-1).item())

print(f"Prompt: {prompt}")
print(f'Prediction: "{prediction}"')

# %% [markdown]
# Well done, BERT!


