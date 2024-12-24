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
import torch
from torch import nn
import torch.nn.functional as F
import math
from sklearn.cluster import KMeans

class Qwen2AudioForRetrieval(Qwen2AudioForConditionalGeneration):
    def token2feature(
        self,
        input_ids: torch.LongTensor = None,
        input_features: torch.FloatTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        feature_attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> torch.FloatTensor:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        target_device = self.audio_tower.device

        if input_features is not None:
            input_features = input_features.to(target_device)
            feature_attention_mask = feature_attention_mask.to(target_device)

        if inputs_embeds is None:
            # 1. Extract the input embeddings
            inputs_embeds = self.get_input_embeddings()(input_ids)

            # 2. Merge text and audios
            if input_features is not None and input_ids.shape[1] != 1:
                audio_feat_lengths, audio_output_lengths = self.audio_tower._get_feat_extract_output_lengths(
                    feature_attention_mask.sum(-1)
                )
                batch_size, _, max_mel_seq_len = input_features.shape
                max_seq_len = (max_mel_seq_len - 2) // 2 + 1
                # Create a sequence tensor of shape (batch_size, max_seq_len)
                seq_range = (
                    torch.arange(0, max_seq_len, dtype=audio_feat_lengths.dtype, device=audio_feat_lengths.device)
                    .unsqueeze(0)
                    .expand(batch_size, max_seq_len)
                )
                lengths_expand = audio_feat_lengths.unsqueeze(1).expand(batch_size, max_seq_len)
                # Create mask
                padding_mask = seq_range >= lengths_expand

                audio_attention_mask_ = padding_mask.view(batch_size, 1, 1, max_seq_len).expand(
                    batch_size, 1, max_seq_len, max_seq_len
                )
                audio_attention_mask = audio_attention_mask_.to(
                    dtype=self.audio_tower.conv1.weight.dtype, device=self.audio_tower.conv1.weight.device
                )
                audio_attention_mask[audio_attention_mask_] = float("-inf")

                audio_outputs = self.audio_tower(input_features, attention_mask=audio_attention_mask)
                selected_audio_feature = audio_outputs.last_hidden_state
                audio_features = self.multi_modal_projector(selected_audio_feature)

                inputs_embeds, attention_mask, labels, position_ids, _ = self._merge_input_ids_with_audio_features(
                    audio_features, audio_output_lengths, inputs_embeds, input_ids, attention_mask, labels
                )

        return inputs_embeds
    
    def generate_caption_prompt(self) -> str:
        """
        Generate a prompt with structured instructions and examples in English.
        """
        instruction = """Please generate a description based on the following music information. The description should include lyrics (whether it has lyrics and what they are), genre, instruments, key, tempo, chords, and emotions, maintaining a consistent structure.
Example:
Lyrics: Yes, content is 'The road of dreams'
Genre: Rock
Instruments: Electric guitar, bass, drums
Key: D Major
Tempo: Allegro
Chords: D-A-Bm-G
Emotions: Exciting, passionate
Description: This is a rock song with lyrics 'The road of dreams', mainly performed by electric guitar, bass, and drums. It is in D Major, with an allegro tempo, chord progression D-A-Bm-G, conveying exciting and passionate emotions.

Now, please generate a music description based on the following information:
"""
        prompt = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nAudio 1: <|audio_bos|><|AUDIO|><|audio_eos|>\n{instruction}<|im_end|>\n<|im_start|>assistant\n"
        return prompt

    def generate_caption(
        self,
        audio_path: List[str],
        processor: AutoProcessor
    ) -> str:
        
        # Load audio
        audio, _ = librosa.load(audio_path, sr=processor.feature_extractor.sampling_rate)

        # 限制音频长度为30s
        sample_per_sec = processor.feature_extractor.sampling_rate
        taget_sec = 30
        audio = audio[:sample_per_sec * taget_sec]
        
        # Prepare inputs
        prompt = self.generate_caption_prompt()
        inputs = processor(text=prompt, audios=audio, return_tensors="pt", sampling_rate=processor.feature_extractor.sampling_rate)
        
        # Generate
        generate_ids = self.generate(**inputs, max_length=5000)
        generate_ids = generate_ids[:, inputs.input_ids.size(1):]
        caption = processor.batch_decode(generate_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)[0]
        return caption
    
    def get_text_features(
        self,
        text: str,
        processor: AutoProcessor
    ):
        inputs = processor(
            text = text,
            audios = None,
            return_tensors="pt",
            padding="max_length",  # Ensure inputs are padded the same way
            max_length=400,
            sampling_rate=processor.feature_extractor.sampling_rate,
        )

        text_features = self.token2feature(**inputs)
        return text_features
    
    def get_audio_features(
        self,
        audio_path: str,
        processor: AutoProcessor
    ):
        # generate audio caption
        audio_caption = self.generate_caption(audio_path, processor)

        # get audio features
        audio_features = self.get_text_features(audio_caption, processor)

        return audio_features
    
def init_music_feature():
    import json
    import os

    # 模型初始化
    model_path = "/home/lr/.cache/modelscope/hub/qwen/Qwen2-Audio-7B-Instruct"
    processor = AutoProcessor.from_pretrained(model_path, device_map="auto")
    transfer_model = Qwen2AudioForRetrieval.from_pretrained(model_path, device_map="auto")

    # 设置路径
    audio_dir = "/home/lr/project/MuChin/dataset/muchin-stats/muchin-audio"
    features_dir = "/home/lr/project/MusicIR/audio_features"
    os.makedirs(features_dir, exist_ok=True)

    # 准备音频路径和映射字典
    audio_feature_mapping = {}
    
    # 处理每个音频文件
    for i in tqdm(range(1, 1001), desc="preprocessing audios", unit="audio"):
        mid = f"s_{str(i).zfill(4)}"
        audio_path = os.path.join(audio_dir, f"{mid}.mp3")
        feature_path = os.path.join(features_dir, f"{mid}.pt")
        
        # 获取并保存特征
        audio_features = transfer_model.get_audio_features(audio_path, processor)
        torch.save(audio_features, feature_path)
        
        # 记录映射关系
        audio_feature_mapping[mid] = {
            "audio_path": audio_path,
            "feature_path": feature_path
        }
    
    # 保存映射关系到 JSON 文件
    mapping_path = "/home/lr/project/MusicIR/audio_feature_mapping.json"
    with open(mapping_path, 'w', encoding='utf-8') as f:
        json.dump(audio_feature_mapping, f, ensure_ascii=False, indent=2)

    # 使用示例:
    """
    # 加载单个特征
    with open('/home/lr/project/MusicIR/audio_feature_mapping.json', 'r') as f:
        mapping = json.load(f)
    
    mid = "s_0001"
    feature_path = mapping[mid]["feature_path"]
    audio_features = torch.load(feature_path)
    """

def init_music_caption():
    # similar to init_music_feature but save caption instead of feature in a json file
    import json
    import os

    # 模型初始化
    model_path = "/home/lr/.cache/modelscope/hub/qwen/Qwen2-Audio-7B-Instruct"
    processor = AutoProcessor.from_pretrained(model_path, device_map="auto")
    transfer_model = Qwen2AudioForRetrieval.from_pretrained(model_path, device_map="auto")

    # 设置路径
    audio_dir = "/home/lr/project/MuChin/dataset/muchin-stats/muchin-audio"
    caption_json_file = "/home/lr/project/MusicIR/audio_caption.json"

    # 准备音频路径和映射字典
    audio_caption_mapping = {}

    # json文件直接保存caption对
    if os.path.exists(caption_json_file):
        with open(caption_json_file, 'r') as f:
            audio_caption_mapping = json.load(f)

    # 处理每个音频文件
    for i in tqdm(range(1, 1001), desc="preprocessing audios", unit="audio"):
        mid = f"s_{str(i).zfill(4)}"
        audio_path = os.path.join(audio_dir, f"{mid}.mp3")
        
        # 获取并保存特征
        audio_caption = transfer_model.generate_caption(audio_path, processor)
        
        # 记录映射关系
        audio_caption_mapping[mid] = audio_caption

    # 保存映射关系到 JSON 文件
    with open(caption_json_file, 'w', encoding='utf-8') as f:
        json.dump(audio_caption_mapping, f, ensure_ascii=False, indent=2)

    # 使用示例:
    """
    # 加载单个特征
    with open('/home/lr/project/MusicIR/audio_caption.json', 'r') as f:
        mapping = json.load(f)

    mid = "s_0001"
    audio_caption = mapping[mid]
    """

def calculate_similarity(query_features, audio_features):
    """计算文本查询和音频特征之间的相似度，并及时释放中间变量显存
    """
    with torch.no_grad():  # 避免存储计算图
        # 1. L2归一化
        query_norm = F.normalize(query_features, p=2, dim=-1)
        audio_norm = F.normalize(audio_features, p=2, dim=-1)
        
        # 2. 重塑维度并计算
        query_reshaped = query_norm.squeeze(0)
        audio_reshaped = audio_norm.view(-1, audio_norm.shape[-1])
        
        # 3. 分块计算注意力分数以减少显存占用
        chunk_size = 32  # 可调整的块大小
        attn_scores_list = []
        
        for i in range(0, audio_reshaped.size(0), chunk_size):
            chunk = audio_reshaped[i:i+chunk_size]
            chunk_score = torch.matmul(query_reshaped, chunk.transpose(0, 1))
            attn_scores_list.append(chunk_score)
            # 即时清理临时变量
            del chunk
            torch.cuda.empty_cache()
            
        # 拼接所有块的结果
        attn_scores = torch.cat(attn_scores_list, dim=1)
        del attn_scores_list
        torch.cuda.empty_cache()
        
        # 重塑注意力分数
        attn_scores = attn_scores.view(query_reshaped.shape[0], -1, audio_norm.shape[1])
        
        # 4. 计算softmax
        scale = math.sqrt(query_reshaped.shape[-1])
        attn_weights = F.softmax(attn_scores / scale, dim=-1)
        del attn_scores
        torch.cuda.empty_cache()
        
        # 5. 计算最终相似度
        sequence_sims = torch.mean(attn_weights, dim=0)
        del attn_weights
        torch.cuda.empty_cache()
        
        similarities = torch.mean(sequence_sims, dim=-1)
        del sequence_sims
        torch.cuda.empty_cache()
        
        # 清理剩余中间变量
        del query_norm, audio_norm, query_reshaped, audio_reshaped
        torch.cuda.empty_cache()
        
        return similarities.detach()  # 确保返回的tensor与计算图分离


def search_related_music(
        dialog_reconstrected_caption: str,
        feature_mapping_json_path: str,
        processor: AutoProcessor,
        model: Qwen2AudioForRetrieval,
        top_k: int = 30,
        kmeans_clusters: int = 5
    ):
    """
    Search for related music based on the reconstructed dialog caption.

    Args:
        dialog_reconstrected_caption (str): The reconstructed dialog caption.
        feature_mapping (dict): The mapping dictionary of audio features.
        processor (AutoProcessor): The processor for the model.
        model (Qwen2AudioForRetrieval): The model for generating captions.
        top_k (int): The number of related music to be returned.

    find top_k related music and than use KMeans to cluster them.
    chose 1 representative music from each cluster.
    """

    # get query text features
    query_text_features = model.get_text_features(dialog_reconstrected_caption, processor)

    # 在文件开头添加设备检测
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    
    # 直接分批处理特征和计算相似度
    batch_size = 256  # 可以调小这个值以减少显存使用
    similarities_list = []
    all_audio_mids = []
    
    with open(feature_mapping_json_path, 'r') as f:
        mapping = json.load(f)
        current_batch = []
        current_batch_mids = []
        
        for mid, paths in mapping.items():
            feature = torch.load(paths["feature_path"])
            # 对特征进行平均池化到固定长度
            if feature.size(1) != 400:
                feature = torch.nn.functional.adaptive_avg_pool2d(
                    feature.transpose(1, 2),  # [1, 4096, length]
                    (4096, 400)  # target size
                ).transpose(1, 2)  # [1, 400, 4096]
            
            current_batch.append(feature)
            current_batch_mids.append(mid)
            
            # 当累积够一个batch时，进行处理
            if len(current_batch) >= batch_size:
                # 堆叠当前批次并计算相似度
                batch_features = torch.stack([f.to(device) for f in current_batch])
                
                # 计算相似度
                batch_similarities = calculate_similarity(
                    query_text_features.to(device),
                    batch_features
                )
                
                # 保存结果并清理
                similarities_list.append(batch_similarities.cpu())
                all_audio_mids.extend(current_batch_mids)
                
                # 清空当前批次并释放显存
                del batch_similarities
                del batch_features
                current_batch = []
                current_batch_mids = []
                torch.cuda.empty_cache()
        
        # 处理最后剩余的样本
        if current_batch:
            batch_features = torch.stack([f.to(device) for f in current_batch])
            batch_similarities = calculate_similarity(
                query_text_features.to(device),
                batch_features
            )
            similarities_list.append(batch_similarities.cpu())
            all_audio_mids.extend(current_batch_mids)
            del batch_features
            del batch_similarities
            torch.cuda.empty_cache()
    
    # 拼接所有结果
    similarities = torch.cat(similarities_list, dim=0)
    
    # get top_k similar music and their features
    top_k_indices = torch.topk(similarities, min(top_k, len(similarities))).indices
    top_k_audio_mids = [all_audio_mids[i] for i in top_k_indices]
    top_k_similarities = similarities[top_k_indices]
    
    # 重新加载top_k音频的特征用于聚类
    top_k_features = []
    for mid in top_k_audio_mids:
        with open(feature_mapping_json_path, 'r') as f:
            mapping = json.load(f)
            feature = torch.load(mapping[mid]["feature_path"])
            # 确保feature不需要计算梯度
            with torch.no_grad():
                if feature.size(1) != 400:
                    feature = torch.nn.functional.adaptive_avg_pool2d(
                        feature.transpose(1, 2),
                        (4096, 400)
                    ).transpose(1, 2)
                # 将特征展平为一维向量用于聚类，添加detach()
                feature = feature.mean(dim=1).squeeze().detach().cpu().numpy()
                top_k_features.append(feature)
    
    # 对top_k个特征进行聚类
    top_k_features = np.stack(top_k_features)
    n_clusters = min(kmeans_clusters, len(top_k_features))
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(top_k_features)
    cluster_labels = kmeans.labels_
    
    # 从每个簇中选择相似度最高的音频作为代表
    representative_mids = []
    for i in range(n_clusters):
        cluster_indices = np.where(cluster_labels == i)[0]
        cluster_similarities = top_k_similarities[cluster_indices]
        representative_index = torch.argmax(cluster_similarities)
        representative_mids.append(top_k_audio_mids[cluster_indices[representative_index]])
    
    return top_k_audio_mids, representative_mids