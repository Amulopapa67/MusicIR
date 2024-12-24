import os
import json
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import Trainer, TrainingArguments, AutoProcessor, Qwen2AudioForConditionalGeneration, Qwen2AudioPreTrainedModel
from transformers import TrainerCallback
from safetensors.torch import load_file
from torch.utils.tensorboard import SummaryWriter
from concurrent.futures import ProcessPoolExecutor
from tqdm.auto import tqdm
from peft import PeftModel, LoraConfig, TaskType
import librosa
from transformers.modeling_outputs import ModelOutput
from transformers.models.qwen2_audio.modeling_qwen2_audio import Qwen2AudioCausalLMOutputWithPast
from typing import Optional, List, Tuple, Union
from torch import nn
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from Qwen2AudioForRetrieval import Qwen2AudioForRetrieval, search_related_music, calculate_similarity
import random

def reconstruct_dialog2caption(
    reconstructed_model: AutoModelForCausalLM,
    reconstructed_tokenizer: AutoTokenizer,  
    dialog: List[Tuple[str, str]]
) -> str:
    # 系统提示
    system_prompt = """You are an AI assistant that reconstructs detailed music descriptions. 
    Given an initial caption and a dialogue about a music piece, generate a comprehensive new caption."""
    
    # 构建当前查询消息
    dialog_text = ", ".join([f"{q}: {a}" for q, a in dialog[1:]])
    current_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Initial caption: {dialog[0][1]}\nDialog: {dialog_text}"}
    ]
    
    # 将消息转换为模型输入格式
    prompt = reconstructed_tokenizer.apply_chat_template(current_messages, tokenize=False)
    inputs = reconstructed_tokenizer(prompt, return_tensors="pt").to(reconstructed_model.device)
    
    outputs = reconstructed_model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.5,
        num_return_sequences=1
    )
    
    # 解码输出并获取第一个换行符后的内容
    decoded_text = reconstructed_tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
    
    # 分割文本并返回第一个换行符后的内容
    split_text = decoded_text.split('\n', 1)
    return split_text[1] if len(split_text) > 1 else decoded_text


def generate_follow_up_questions_list(
    questioner_model: AutoModelForCausalLM,
    questioner_tokenizer: AutoTokenizer,
    description: str,
    candidates: List[str],
    dialog_history: List[Tuple[str, str]] = None
) -> List[str]:
    """
    Generate multiple follow-up questions about the target music piece based on prior dialogue.
    
    Args:
        questioner_model: The question generation model.
        questioner_tokenizer: The tokenizer for the question generation model.
        description: The description of the target music piece.
        candidates: A list of retrieval candidates.
        dialog_history: The dialogue history.
    Returns:
        List[str]: A list of generated follow-up questions
    """
    system_prompt = """You are a proficient question generator tasked with generating multiple distinct questions for music retrieval.
    Analyze the description, candidates, and previous dialogue to generate 3-5 new questions that explore aspects not yet discussed.
    Each question should start with "Q: " and be on a new line.
    
    Example input:
    Description: A slow piano ballad with emotional vocals
    
    Previous Dialog:
    Q: What is the tempo of the song?
    A: The song has a slow tempo
    
    Example output:
    Q: Does the piano play any memorable melodic patterns?
    Q: How would you describe the emotional tone of the vocals?
    Q: Are there any other instruments besides the piano?
    Q: Does the song build up in intensity or maintain the same mood throughout?"""
    
    user_content = f"Description: {description}\n\nRetrieval Candidates:\n"
    user_content += "\n".join([f"{idx}. {candidate}" for idx, candidate in enumerate(candidates)])
    
    if dialog_history:
        user_content += "\n\nPrevious Dialog:\n"
        user_content += "\n".join([f"Q: {q}\nA: {a}" for q, a in dialog_history])
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content}
    ]
    
    prompt = questioner_tokenizer.apply_chat_template(messages, tokenize=False)
    inputs = questioner_tokenizer(prompt, return_tensors="pt").to(questioner_model.device)
    
    outputs = questioner_model.generate(
        **inputs,
        max_new_tokens=256,
        temperature=0.7,
        num_return_sequences=1,
        do_sample=True
    )
    
    generated_text = questioner_tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
    
    # 将生成的文本分割成问题列表
    questions = [q.strip()[3:] for q in generated_text.split('\n') if q.strip().startswith('Q:')]
    
    return questions

def select_question(
    dialog_history: List[Tuple[str, str]],
    questions: List[str],
    model: Qwen2AudioForRetrieval,
    processor: AutoProcessor,
) -> str:
    """
    Select the most diverse question from candidates based on current dialog context
    
    Args:
        dialog_history: List of (question, answer) tuples representing conversation history
        questions: List of candidate questions
        model: Qwen2Audio model instance
        processor: Model processor
        
    Returns:
        str: Selected question that maximizes information coverage
    """
    # 构建对话历史上下文字符串
    context = ""
    if dialog_history:
        context = " ".join([f"Q: {q} A: {a}" for q, a in dialog_history])
    
    # 获取上下文特征
    context_features = None
    if context:
        context_features = model.get_text_features(context, processor)  # shape: (1, seq_len, dim)
    
    # 如果没有上下文，随机选择问题
    if context_features is None:
        return random.choice(questions)
     
    # 获取所有问题的特征
    question_features = []
    for question in questions:
        features = model.get_text_features(question, processor)  # shape: (1, seq_len, dim)
        question_features.append(features)
    
    # 将问题特征堆叠成批处理形式
    question_features = torch.cat(question_features, dim=0)  # shape: (n_questions, seq_len, dim)
    
    # 计算上下文与所有问题之间的相似度
    similarities = calculate_similarity(context_features, question_features)  # shape: (n_questions,)
    
    # 选择相似度最低的问题
    selected_idx = similarities.argmin().item()
    
    return questions[selected_idx]

