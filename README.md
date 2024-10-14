# MLLM Can See? Dynamic Correction Decoding For Hallucination Mitigation

<!-- [![License: MIT](https://img.shields.io/badge/License-MIT-g.svg)](https://opensource.org/licenses/MIT)
[![Arxiv](https://img.shields.io/badge/arXiv-2311.17911-B21A1B)](https://arxiv.org/pdf/2311.17911.pdf)
[![Hugging Face Transformers](https://img.shields.io/badge/%F0%9F%A4%97-Transformers-blue)](https://github.com/huggingface/transformers)
[![GitHub Stars](https://img.shields.io/github/stars/shikiw/OPERA?style=social)](https://github.com/shikiw/OPERA/stargazers) -->


This repository provides the official PyTorch implementation of the following paper: 
> [**MLLM Can See? Dynamic Correction Decoding For Hallucination Mitigation**](https://arxiv.org/pdf/2311.17911.pdf) <br>
> Chenxi Wang<sup>1</sup>, 
> Xiang Chen<sup>1</sup>, 
> Ningyu Zhang<sup>1</sup>,
> Bozhong Tian<sup>1</sup>,
> Haoming Xu<sup>1</sup>, 
> Shumin Deng<sup>2</sup>,
> Huajun Chen<sup>1</sup> <br>
> <sup>1</sup>Zhejiang University, <sup>2</sup>National University of Singapore <br>


## Overview

<p align="center"><img src="img/method.png" alt="teaser" width="500px" /></p>

Multimodal Large Language Models (MLLMs) frequently exhibit hallucination phenomena, but the underlying reasons remain poorly understood. In this paper, we present an empirical analysis and find that, although MLLMs incorrectly generate the targets in the final output, they are actually able to recognize visual objects in the preceding layers. We speculate that this may be due to the strong knowledge priors of the language model suppressing the visual information, leading to hallucinations. Motivated by this, we propose a novel dynamic correction decoding method for MLLMs (DeCo), which adaptively selects the appropriate preceding layers and proportionally integrates knowledge into the final layer to adjust the output logits. Note that DeCo is model agnostic and can be seamlessly incorporated with various classic decoding strategies and applied to different MLLMs. We evaluate DeCo on widely-used benchmarks, demonstrating that it can reduce hallucination rates by a large margin compared to baselines, highlighting its potential to mitigate hallucinations.

## Setup

The main implementation of Deco is in `transformers/generation/utils.py`.

```
conda create -n deco python==3.9
conda activate deco
pip install -r requirements.txt
```

## TL;DR
After setup the environment, you can directly use Deco on your own MLLM model by:
```
with torch.no_grad():
    output_dict = model.generate(
                    input_ids,
                    images=image_tensor,
                    do_sample=args.do_sample
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    max_new_tokens=args.max_new_tokens,
                    return_dict_in_generate=True,
                    output_hidden_states=True,
                    use_deco = True,
                    alpha = 0.6,
                    threshold_top_p = 0.9, 
                    threshold_top_k = 20,
                    early_exit_layers=[i for i in range(20, 29)]
                    )
                
output_ids = output_dict.sequences
outputs = tokenizer.batch_decode(output_ids)
```
<!-- 
Please refer to `demo.ipynb` [here](https://github.com/shikiw/OPERA/blob/1e74d8b5d082579c81e0e77ef1cf4a44d20ab91e/demo.ipynb) for more details. -->


## Evaluation

The following evaluation requires for MSCOCO 2014 dataset. Please download [here](https://cocodataset.org/#home) and extract it in your data path.

<!-- Besides, it needs you to prepare the following checkpoints of 7B base models: -->

<!-- - Download [LLaVA-1.5 merged 7B model](https://huggingface.co/liuhaotian/llava-v1.5-7b) and specify it at [Line 14](https://github.com/shikiw/OPERA/blob/bf18aa9c409f28b31168b0f71ebf8457ae8063d5/eval_configs/llava-1.5_eval.yaml#L14) of `eval_configs/llava-1.5_eval.yaml`.
- Download [Vicuna 7B v1.1 model](https://github.com/lm-sys/FastChat) and specify it at [Line 25](https://github.com/shikiw/OPERA/blob/bf18aa9c409f28b31168b0f71ebf8457ae8063d5/minigpt4/configs/models/blip2_instruct_vicuna7b.yaml#L25) of `minigpt4/configs/models/blip2_instruct_vicuna7b.yaml`.
- Download [Vicuna 7B v0 model](https://huggingface.co/Vision-CAIR/vicuna-7b/tree/main) and specify it at [Line 18](https://github.com/shikiw/OPERA/blob/bf18aa9c409f28b31168b0f71ebf8457ae8063d5/minigpt4/configs/models/minigpt4_vicuna0.yaml#L18) of `minigpt4/configs/models/minigpt4_vicuna0.yaml`.
- Download [MiniGPT-4 7B pretrained weights](https://drive.google.com/file/d/1RY9jV0dyqLX-o38LrumkKRh6Jtaop58R/view?usp=sharing) and specify it at [Line 8](https://github.com/shikiw/OPERA/blob/bf18aa9c409f28b31168b0f71ebf8457ae8063d5/eval_configs/minigpt4_eval.yaml#L8) of `eval_configs/minigpt4_eval.yaml`.
- Download [Shikra merged 7B model](https://github.com/shikras/shikra#checkpoint) and specify it at [Line 14](https://github.com/shikiw/OPERA/blob/bf18aa9c409f28b31168b0f71ebf8457ae8063d5/eval_configs/shikra_eval.yaml#L14) of `eval_configs/shikra_eval.yaml`. -->

### Arguments

| Argument             | Example             | Description   |
| -------------------- | ------------------- | ------------- |
| `--model`    | `llava-1.5` | Specify the MLLM model, this codebase supports `instructblip`, `minigpt4`, `llava-1.5`, `shikra`. |
| `--data-path`     | `/path/to/dataset` | Path to the dataset file or folder, e.g., `COCO_2014/val2014/`. |
| `--alpha`   | `0.5` | The scale factor to scale up the calibration strength.. |
| `--threshold_top_p`      | `0.9` | The threshold for controlling the number of candidate tokens. |
| `--early-exit-layers`   | `range(20,29)` | The candidate layer interval can be adjusted appropriately according to the model. |


### CHAIR
- Generate the MLLM's responses and save them in a jsonl file:
```bash
python chair_llava.py
```
<!-- Note: Please check out our released results in `log/chair_eval_results` for reproduction. -->

- Calculate CHAIR using the generated jsonl file:
```bash
python chair.py --cap_file /path/to/jsonl --image_id_key image_id --caption_key caption --coco_path /path/to/COCO/annotations_trainval2014/annotations/ --save_path /path/to/save/jsonl
```


### POPE
```bash
python pope_eval.py 
```
### MME
```bash
python mme_llava.py
```




## Reference Repositories
- DoLa: https://github.com/voidism/DoLa
- OPERA: https://github.com/shikiw/OPERA
- VCD: https://github.com/DAMO-NLP-SG/VCD
- LLaVA: https://github.com/haotian-liu/LLaVA


<!-- ## Acknowledgement
This repo is based on the MLLM codebase of [LAVIS](https://github.com/salesforce/LAVIS) and [MiniGPT-4](https://github.com/Vision-CAIR/MiniGPT-4) and the CHAIR code of [Maxlinn](https://github.com/Maxlinn/CHAIR-metric-standalone). Thanks for their impressive works! -->

## Citation
If you find this work useful for your research, please cite [our paper](https://arxiv.org/pdf/2311.17911.pdf):
```
@inproceedings{huang2024opera,
  title={MLLM Can See? Dynamic Correction Decoding For Hallucination Mitigation},
  author={Chenxi Wang, Xiang Chen, Ningyu Zhang, Bozhong Tian, Haoming Xu, Shumin Deng, Huajun Chen},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={13418--13427},
  year={2024}
}
```