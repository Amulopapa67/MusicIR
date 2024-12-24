<div align="center">
  <img src="assets/logo.webp" alt="MusicIR Logo" width="200"/>
</div>

# Music IR System

This repository provides tools for building and running a music information retrieval system. Users can construct their own audio databases, generate corresponding caption and feature files, and run a retrieval demo. Note that due to copyright reasons, audio files are not publicly provided.

## Installation

1. Create and activate a new conda environment (recommended):
```bash
conda create -n music_ir python=3.10
conda activate music_ir
```

2. Install PyTorch first (for CUDA 11.8):
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

3. Install other requirements:
```bash
pip install -r requirements.txt
```

## Project Structure

- `retrievalDemo.py`: Web interface for music retrieval
- `retrievalFunction.py`: Core functions for music retrieval
- `Qwen2AudioForRetrieval.py`: Modified Qwen2Audio model for feature extraction and caption generation

## Usage

### Step 1: Construct Your Audio Dataset

To construct your dataset, follow these steps:

1. Generate Caption and Feature Files
   - Use the functions `init_music_caption()` and `init_music_feature()` in `Qwen2AudioForRetrieval.py`
   - Modify the audio file paths inside these functions according to your dataset
   - Running these functions will generate:
     - A feature directory (`features/`)
     - A JSON file mapping audio files to captions and features

Example Workflow:
```python
# Inside Qwen2AudioForRetrieval.py
init_music_caption()
init_music_feature()
```

### Step 2: Run the Demo

Start the web interface for music retrieval:
```bash
python retrievalDemo.py
```

## Citation

If you use this code or its components in your research, please cite the following works:

**Qwen2.5 Technical Report:**
```bibtex
@article{qwen2.5,
    title   = {Qwen2.5 Technical Report}, 
    author  = {An Yang and Baosong Yang and Beichen Zhang and Binyuan Hui and Bo Zheng and Bowen Yu and Chengyuan Li and Dayiheng Liu and Fei Huang and Haoran Wei and Huan Lin and Jian Yang and Jianhong Tu and Jianwei Zhang and Jianxin Yang and Jiaxi Yang and Jingren Zhou and Junyang Lin and Kai Dang and Keming Lu and Keqin Bao and Kexin Yang and Le Yu and Mei Li and Mingfeng Xue and Pei Zhang and Qin Zhu and Rui Men and Runji Lin and Tianhao Li and Tingyu Xia and Xingzhang Ren and Xuancheng Ren and Yang Fan and Yang Su and Yichang Zhang and Yu Wan and Yuqiong Liu and Zeyu Cui and Zhenru Zhang and Zihan Qiu},
    journal = {arXiv preprint arXiv:2412.15115},
    year    = {2024}
}
```

**Qwen2 Technical Report:**
```bibtex
@article{qwen2,
    title   = {Qwen2 Technical Report}, 
    author  = {An Yang and Baosong Yang and Binyuan Hui and Bo Zheng and Bowen Yu and Chang Zhou and Chengpeng Li and Chengyuan Li and Dayiheng Liu and Fei Huang and Guanting Dong and Haoran Wei and Huan Lin and Jialong Tang and Jialin Wang and Jian Yang and Jianhong Tu and Jianwei Zhang and Jianxin Ma and Jin Xu and Jingren Zhou and Jinze Bai and Jinzheng He and Junyang Lin and Kai Dang and Keming Lu and Keqin Chen and Kexin Yang and Mei Li and Mingfeng Xue and Na Ni and Pei Zhang and Peng Wang and Ru Peng and Rui Men and Ruize Gao and Runji Lin and Shijie Wang and Shuai Bai and Sinan Tan and Tianhang Zhu and Tianhao Li and Tianyu Liu and Wenbin Ge and Xiaodong Deng and Xiaohuan Zhou and Xingzhang Ren and Xinyu Zhang and Xipin Wei and Xuancheng Ren and Yang Fan and Yang Yao and Yichang Zhang and Yu Wan and Yunfei Chu and Yuqiong Liu and Zeyu Cui and Zhenru Zhang and Zhihao Fan},
    journal = {arXiv preprint arXiv:2407.10671},
    year    = {2024}
}
```

**Qwen2-Audio Technical Report:**
```bibtex
@article{Qwen2-Audio,
    title={Qwen2-Audio Technical Report},
    author={Chu, Yunfei and Xu, Jin and Yang, Qian and Wei, Haojie and Wei, Xipin and Guo,  Zhifang and Leng, Yichong and Lv, Yuanjun and He, Jinzheng and Lin, Junyang and Zhou, Chang and Zhou, Jingren},
    journal={arXiv preprint arXiv:2407.10759},
    year={2024}
}
```

**PlugIR Repository:**
```bibtex
@misc{plugir,
    author = {Saehyung Lee},
    title = {PlugIR: A retrieval library},
    year = {2023},
    publisher = {GitHub},
    journal = {GitHub repository},
    url = {https://github.com/Saehyung-Lee/PlugIR}
}
```

## License

This project is licensed under the MIT License. See the LICENSE file for details.