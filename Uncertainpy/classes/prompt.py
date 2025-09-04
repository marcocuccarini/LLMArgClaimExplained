import os
import sys
import json
import pickle
import re
import networkx as nx
import pandas as pd
from tqdm import tqdm

import ollama
from huggingface_hub import hf_hub_download
from sklearn.metrics import classification_report, f1_score, accuracy_score



# =============== PROMPT FOR ARGUMENT GRAPH CONSTRUCTION========================
EV2C_PROMPT = """
Task: Given a claim and multiple pieces of evidence, classify each evidence as "support", "contradict", or "irrelevant" to the claim.
Instructions:
- Support: Evidence that backs the claim.
- Contradict: Evidence that counters or limits the claim.
- Irrelevant: Evidence unrelated to the claim.
Output Format:
Return a single JSON object with three keys: "support", "contradict", and "irrelevant", each mapping to a list of evidence items.
Example format: {example}
Claim:
{claim}
Evidence:
{evidence}
You must always PROVIDE ONLY A SINGLE JSON without any additional explanation or commentary.
**Do not** include markdown formatting in the output.
"""
EV2C_JSON_EXAMPLE = '{"support": ["E1"], "contradict": ["E3", "E4"], "irrelevant": ["E2", "E5"]}'

EV2EV_PROMPT = """
Task: Given a claim and multiple pieces of evidence, analyze the relationships between evidence with respect to the claim.
Instructions:
- Support: Two evidence items that reinforce each other regarding the claim.
- Contradict: Two evidence items that conflict with each other regarding the claim.
Output Format:
Return a single JSON object with two keys: "support" and "contradict", each mapping to a list of evidence pairs.
Example format: {example}
Claim:
{claim}
Evidence:
{evidence}
You must always PROVIDE ONLY A SINGLE JSON without any additional explanation or commentary.
**Do not** include markdown formatting in the output.
"""
EV2EV_JSON_EXAMPLE = '{"support": [["E1", "E2"], ["E1", "E3"]], "contradict": [["E2", "E3"]]}'
# ===============================================

# classes/prompt.py

# ---------------------------
# PROMPTS
# ---------------------------

# classes/prompt.py

# ---------------------------
# PROMPTS
# ---------------------------

PROMPT_BASELINE_RETRIVER = """
You are a fact-checking expert.
For each input claim, output only true if the claim is factually correct,
or false if it is not. Respond with a single word (true or false) | no
explanations, justifications, or additional text.
Claim: {claim} 
Answer:"""

PROMPT_BASELINE_IC_RALM = """
You are a fact-checking expert.
Given a claim and retrieved evidence, output true if the claim is factually
supported, or false if it is not. Base your answer only on the provided
evidence and your own knowledge if necessary. Respond with a single word
(true or false) | no explanations, reasoning, or additional text.
Claim: {claim} Evidence: {evidence} Answer:
"""

PROMPT_BASELINE_IC_RALM_P = """
You are a fact-checking expert.
Given a claim and retrieved evidence, output true if the claim is factually
supported, or false if it is not. Base your answer primarily on the provided
evidence, using your own knowledge only if necessary. Consider only evidence
that is directly relevant to the claim. Respond with a single word (true or
false) | no explanations, reasoning, or additional text.
Claim: {claim} Evidence: {evidence} Answer:
"""

PROMPT_BASELINE_EXP = """
You are a fact-checking expert.
Your task is to evaluate the truthfulness of a claim based on the provided
evidence. You must provide a score from 0 to 100, where 0 represents
definitively False and 100 represents clearly True. Your score should reflect
your assessment of the claim’s truthfulness in relation to the evidence.
Return a JSON object with two keys: first, your analysis in the "explanation"
key, then a comma, finally a score with the "score" key. The score should
match the analysis and your assessment of the claim’s truthfulness.
Claim: {claim} Evidence: {evidence}
OUTPUT FORMAT = {{"explanation": "<explanation>", "score": <score>}}
"""

PROMPT_BASELINE_CoT = """
You are a fact-checking expert.
Your task is to evaluate the truthfulness of a claim based on the provided
evidence. You must provide a score from 0 to 100, where 0 represents
definitively False and 100 represents clearly True. Your score should reflect
your assessment of the claim’s truthfulness in relation to the evidence.
Return a JSON object with two keys: first, your analysis in the "explanation"
key, then a comma, finally a score with the "score" key. The score should
match the analysis and your assessment of the claim’s truthfulness.
Claim: {claim} Evidence: {evidence}
OUTPUT FORMAT = {{"explanation": "<explanation>", "score": <score>}}
"""

# ---------------------------
# Placeholder for grad.Argument if needed
# ---------------------------
class Argument:
    def __init__(self, *args, **kwargs):
        pass

# ---------------------------
# Prompt Registry
# ---------------------------
FACTCHECK_PROMPTS = {
    "baseline_retriever": (PROMPT_BASELINE_RETRIVER, "binary"),
    "baseline_ic_ralm": (PROMPT_BASELINE_IC_RALM, "binary"),
    "baseline_ic_ralm_p": (PROMPT_BASELINE_IC_RALM_P, "binary"),
    "baseline_exp": (PROMPT_BASELINE_EXP, "json_score"),
    "baseline_cot": (PROMPT_BASELINE_CoT, "json_score"),
}
