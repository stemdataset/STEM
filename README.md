# Measuring Vision-Language STEM Skills of Neural Models
The code for ICLR 2024 paper: [Measuring Vision-Language STEM Skills of Neural Models](https://arxiv.org/abs/2402.17205).
<p align="center">
  📃 <a href="https://arxiv.org/abs/2402.17205" target="_blank">[Paper]</a> • 💻 <a href="https://github.com/stemdataset/STEM" target="_blank">[Github]</a> • 🤗 <a href="https://huggingface.co/datasets/stemdataset/STEM" target="_blank">[Dataset]</a> • 🏆 <a href="https://huggingface.co/spaces/stemdataset/stem-leaderboard" target="_blank">[Leaderboard]</a> • 📽 <a href="https://github.com/stemdataset/STEM/blob/main/assets/STEM-Slides.pdf" target="_blank">[Slides]</a> • 📋 <a href="https://github.com/stemdataset/STEM/blob/main/assets/poster.pdf" target="_blank">[Poster]</a>
</p>

## Setup Environment
We recommend using Anaconda to create a new environment and install the required packages. You can create a new environment and install the required packages using the following commands:
```bash
conda create -n clip python=3.10
conda activate clip
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 -c pytorch
pip install git+https://github.com/openai/CLIP.git
pip install transformers==4.18.0
pip install datasets
```

## Run the Code
You can run the inference code using the following command:
```bash
bash run_eval_clip.sh ${eval_split}
```
where `${eval_split}` is the evaluation split you want to evaluate on. The evaluation splits are `valid`, `test`. The results will be saved in `results/clip_${model}_${eval_split}/`. You can submit the `preds.txt` to the leaderboard for the `test` split evaluation.

## Citation
```bibtex
@inproceedings{shen2024measuring,
  title={Measuring Vision-Language STEM Skills of Neural Models},
  author={Shen, Jianhao and Yuan, Ye and Mirzoyan, Srbuhi and Zhang, Ming and Wang, Chenguang},
  booktitle={ICLR},
  year={2024}
}
```
