import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# print(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path, KeywordsStoppingCriteria

from PIL import Image
import math
import torch.distributed as dist
from utils import dist_util
from utils.logger import create_logger
from glob import glob

# import kornia
from transformers import set_seed
# from avisc_utils.vcd_add_noise import add_diffusion_noise
# from avisc_utils.avisc_sample import evolve_avisc_sampling
# evolve_avisc_sampling()

def recorder(out):
    NEG_WORDS = ["No", "not", "no", "NO"]

    out = out.replace('.', '')
    out = out.replace(',', '')
    words = out.split(' ')
    if any(word in NEG_WORDS for word in words) or any(word.endswith("n't") for word in words):
        return "No"
    else:
        return "Yes"


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def eval_model(args):
    
    # set up gpu and logging
    
    dist_util.setup_dist(args)
    device = dist_util.device()

    # Setup an experiment folder:
    if dist.get_rank() == 0:
        os.makedirs(
            args.log_path, exist_ok=True
        )  # Make results folder (holds all experiment subfolders)
        model_string_name = args.model_path.split("/")[-1]
        experiment_index = len(glob(f"{args.log_path}/{model_string_name}/*"))
        experiment_dir = f"{args.log_path}"  # Create an experiment folder
        os.makedirs(experiment_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
    else:
        logger = create_logger(None)
    # logger.info(f"use_cd: {args.use_cd}, method: {args.use_avisc}, layer_gamma: {args.layer_gamma}, masking_scheme: {args.masking_scheme}, lamb: {args.lamb}")
    logger.info(f"question_file : {args.question_file}")


    
    
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name,device="cuda:0",device_map="cuda:0")

    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    # questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    for line in tqdm(questions):
    # for (input_ids, image_tensor, image_sizes), line in tqdm(zip(data_loader, questions), total=len(questions)):
        idx = line["question_id"]
        image_file = line["image"]
        qs = line["text"]
        cur_prompt = qs
        # one word processing
        qs = qs.split('\n')[0] 

        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
        # conv.append_message(conv.roles[0], qs + " Please answer this question with one word.")
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda(0)

        image = Image.open(os.path.join(args.image_folder, image_file))
        image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0] 
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
        with torch.inference_mode():
            output_dict = model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).half().cuda(0),
                do_sample=True if args.temperature>0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                max_new_tokens=1024,
                use_deco=True,
                alpha = args.alpha,
                threshold_top_p=args.threshold_top_p, 
                threshold_top_k=args.threshold_top_k,
                early_exit_layers=[i for i in range(args.start_layer, args.end_layer)],
                output_hidden_states=True,
                return_dict_in_generate=True,
                stopping_criteria=[stopping_criteria],
            )
        output_ids = output_dict.sequences
        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()
        logger.info(f"[{image_file}]")
        logger.info(f"prompt: {cur_prompt}") 
        logger.info(f"text: {outputs}")  
        outputs = recorder(outputs)          

        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": cur_prompt,
                                   "text": outputs,
                                   "model_id": model_name,
                                   "image": image_file,
                                   "metadata": {}}) + "\n")
        ans_file.flush()
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default=".../llava-v1.5-7b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default=".../MME_Benchmark_release_version")
    parser.add_argument("--question-file", type=str, default="/MME/llava_hallu_mme.jsonl")
    parser.add_argument("--answers-file", type=str, default="")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=None)
    
    parser.add_argument("--log_path", type=str, default=".../code/DoLa_MLLM")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--noise_step", type=int, default=500)
    parser.add_argument("--use_cd", type=str2bool, default=False)
    parser.add_argument("--cd_alpha", type=float, default=1)
    parser.add_argument("--cd_beta", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    
    parser.add_argument("--alpha", type=float, default=0.6)
    parser.add_argument("--threshold_top_p", type=float, default=0.9)
    parser.add_argument("--threshold_top_k", type=int, default=20)
    parser.add_argument("--start_layer", type=int, default=20)
    parser.add_argument("--end_layer", type=int, default=29)

    args = parser.parse_args()
    set_seed(args.seed)
    eval_model(args)
