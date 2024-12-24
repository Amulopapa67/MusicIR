import gradio as gr
import json
import os
from typing import List, Tuple
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
from Qwen2AudioForRetrieval import Qwen2AudioForRetrieval, search_related_music
import random
import retrievalFunction as rf

class MusicSearchInterface:
    def __init__(self):
        # 初始化模型
        self.model_path = "/home/lr/.cache/modelscope/hub/qwen/Qwen2-Audio-7B-Instruct"
        self.processor = AutoProcessor.from_pretrained(self.model_path, device_map="auto")
        self.model = Qwen2AudioForRetrieval.from_pretrained(self.model_path, device_map="auto")
        
        # 加载问答模型
        self.questioner_model = AutoModelForCausalLM.from_pretrained("/home/lr/.cache/modelscope/hub/Qwen/Qwen2___5-3B-Instruct", device_map="auto")
        self.questioner_tokenizer = AutoTokenizer.from_pretrained("/home/lr/.cache/modelscope/hub/Qwen/Qwen2___5-3B-Instruct", device_map="auto")
        
        # 其他初始化
        self.audio_dir = "/home/lr/project/MuChin/dataset/muchin-stats/muchin-audio"
        self.feature_mapping_path = "/home/lr/project/MusicIR/audio_feature_mapping.json"
        
        # 状态变量
        self.dialog_history = []
        self.current_candidates = []
        self.round_count = 0
        
    def init_chat(self, initial_description: str):
        """初始化对话"""
        self.dialog_history = [("Initial Description", initial_description)]
        self.round_count = 0
        # 生成第一轮检索结果
        caption = rf.reconstruct_dialog2caption(
            self.questioner_model, 
            self.questioner_tokenizer,
            self.dialog_history
        )
        top_k, representatives = search_related_music(
            caption,
            self.feature_mapping_path,
            self.processor,
            self.model
        )
        self.current_candidates = representatives
        
        # 生成问题列表
        questions = rf.generate_follow_up_questions_list(
            self.questioner_model,
            self.questioner_tokenizer,
            caption,
            representatives,
            self.dialog_history
        )
        
        next_question = rf.select_question(
            self.dialog_history,
            questions,
            self.model,
            self.processor
        )
        
        # 准备音频路径列表
        audio_paths = [os.path.join(self.audio_dir, f"{mid}.mp3") for mid in representatives]
        
        return next_question, audio_paths, self.round_count
    
    def chat_step(self, user_answer: str, current_question: str):
        """处理用户回答并生成下一个问题"""
        self.round_count += 1
        self.dialog_history.append((current_question, user_answer))
        
        # 重构caption
        caption = rf.reconstruct_dialog2caption(
            self.questioner_model,
            self.questioner_tokenizer,
            self.dialog_history
        )
        
        # 重新检索
        top_k, representatives = search_related_music(
            caption,
            self.feature_mapping_path,
            self.processor,
            self.model
        )
        self.current_candidates = representatives
        
        # 生成新问题
        questions = rf.generate_follow_up_questions_list(
            self.questioner_model,
            self.questioner_tokenizer,
            caption,
            representatives,
            self.dialog_history
        )
        
        next_question = rf.select_question(
            self.dialog_history,
            questions,
            self.model,
            self.processor
        )
        
        # 准备音频路径列表
        audio_paths = [os.path.join(self.audio_dir, f"{mid}.mp3") for mid in representatives]
        
        return next_question, audio_paths, self.round_count

def create_interface():
    search_interface = MusicSearchInterface()
    
    with gr.Blocks() as demo:
        gr.Markdown("# 音乐检索对话系统")
        
        with gr.Column():
            initial_input = gr.Textbox(label="请描述您想要找的音乐")
            start_btn = gr.Button("开始检索", size="sm")
            
            chat_history = gr.Chatbot(label="对话历史", height=400)
            current_caption = gr.Textbox(label="当前检索描述", lines=3)
            
            with gr.Row():
                answer_input = gr.Textbox(label="您的回答", scale=4)
                with gr.Column(scale=1):
                    next_btn = gr.Button("继续", size="sm")
                    restart_btn = gr.Button("重新开始", size="sm")
            
            gr.Markdown("### 候选音乐")
            with gr.Row():
                candidates = []
                for i in range(5):
                    with gr.Column(scale=1):
                        audio = gr.Audio(label=f"候选 {i+1}", interactive=False)
                        mid_text = gr.Textbox(label=f"ID {i+1}", show_label=False, 
                                            max_lines=1, container=False)
                        candidates.append((audio, mid_text))

        def on_start(initial_desc):
            if not initial_desc.strip():
                return gr.update()
            q, audios, round_num = search_interface.init_chat(initial_desc)
            
            caption = rf.reconstruct_dialog2caption(
                search_interface.questioner_model,
                search_interface.questioner_tokenizer,
                search_interface.dialog_history
            )
            
            chat_hist = [(initial_desc, None), (q, None)]
            audio_outputs = []
            for audio, mid in zip(audios, search_interface.current_candidates):
                audio_outputs.extend([audio, str(mid)])
            while len(audio_outputs) < 10:  # 补充空值到5对
                audio_outputs.extend([None, ""])
                
            return [chat_hist, caption] + audio_outputs

        def on_next(answer, history):
            if not answer.strip():
                return gr.update()
            
            # 获取当前问题(已经在历史记录中)
            current_q = history[-1][0] if history else ""
            
            # 执行对话步骤
            q, audios, round_num = search_interface.chat_step(answer, current_q)
            
            # 获取新的caption
            caption = rf.reconstruct_dialog2caption(
                search_interface.questioner_model,
                search_interface.questioner_tokenizer,
                search_interface.dialog_history
            )
            
            # 更新对话历史 - 只添加用户回答和新问题
            # 移除最后一个空回答的条目
            history = history[:-1]  # 移除最后一个未回答的问题
            history.append((current_q, answer))  # 添加用户回答
            history.append((q, None))  # 添加新问题
            
            # 准备音频输出
            audio_outputs = []
            for audio, mid in zip(audios, search_interface.current_candidates):
                audio_outputs.extend([audio, str(mid)])
            while len(audio_outputs) < 10:
                audio_outputs.extend([None, ""])
                
            return [history, caption, ""] + audio_outputs

        def on_restart():
            search_interface.dialog_history = []
            search_interface.round_count = 0
            return [[], "", ""] + [None, ""] * 5

        start_btn.click(
            on_start,
            inputs=[initial_input],
            outputs=[chat_history, current_caption] + 
                    [item for pair in candidates for item in pair]
        )
        
        next_btn.click(
            on_next,
            inputs=[answer_input, chat_history],
            outputs=[chat_history, current_caption, answer_input] + 
                    [item for pair in candidates for item in pair]
        )
        
        restart_btn.click(
            on_restart,
            outputs=[chat_history, current_caption, answer_input] + 
                    [item for pair in candidates for item in pair]
        )
    
    return demo

if __name__ == "__main__":
    demo = create_interface()
    demo.launch(share=True)