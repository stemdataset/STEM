import clip
import torch
import pathlib
import torchvision
import argparse
import os
import pickle
import json
import io
from datasets import load_dataset, DatasetDict
from tqdm import tqdm
from torchvision.transforms import Compose, Normalize
from PIL import Image
from src.utils import bytes_to_image, CLIPWrapper, collate_fn, get_data_point
from dataclasses import dataclass


@dataclass
class ResultLogger:
    total: int
    correct: int
    preds: list[int]


def run_eval(
    model: CLIPWrapper, stem_dataset: DatasetDict, eval_split: str, output_dir: str
):
    os.makedirs(output_dir, exist_ok=True)
    result_logger = ResultLogger(0, 0, [])
    for data_item in tqdm(stem_dataset[eval_split], ascii=True, desc="Evaluating"):
        tensor_data = get_data_point(data_item)
        batch_data = collate_fn(
            [tensor_data], model.max_choice, model.pic_size, model.processor
        )
        with torch.no_grad():
            text_input = batch_data["text_info"].cuda()  # batch*MAX_CHOICE, 77
            image_input = batch_data["pic_info"].cuda()
            logits_fix = batch_data["pad_mask"].cuda()
            logits_per_pair = model(text_input, image_input)
            logits_per_pair = logits_per_pair + logits_fix

            predicted = torch.argmax(logits_per_pair, dim=1).cpu()
            predicted = predicted.item()
            answer_id = batch_data["answer_id"].item()
            if eval_split == "test":
                result_logger.total += 1
                result_logger.preds.append(predicted)
                assert answer_id == -1
            elif eval_split == "valid":
                result_logger.total += 1
                result_logger.correct += int(predicted == answer_id)
                result_logger.preds.append(predicted)
            else:
                raise ValueError(f"Invalid eval_split: {eval_split}")
    with open(os.path.join(output_dir, "preds.txt"), "w") as f:
        f.write("\n".join(map(str, result_logger.preds)))
    with open(os.path.join(output_dir, "result.pkl"), "wb") as f:
        pickle.dump(result_logger, f)
    print("Saved predictions to {}.".format(os.path.join(output_dir, "preds.txt")))
    print(
        "Total/Correct/ACC: {}/{}/{:.2f}%".format(
            result_logger.total,
            result_logger.correct,
            result_logger.correct / result_logger.total * 100,
        )
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="ViT-B/32", type=str)
    parser.add_argument("--load_model_dir", default=None, type=str)
    parser.add_argument("--eval_split", choices=["test", "valid"], default="valid")
    parser.add_argument("--output_dir", default="results/debug-output/", type=str)
    parser.add_argument("--pic_size", default=224, type=int)
    args = parser.parse_args()
    print("Args:", json.dumps(vars(args), indent=2))
    stem_dataset = load_dataset("stemdataset/STEM")
    max_choice = -1
    for data_item in stem_dataset[args.eval_split]:
        if data_item["pic_choice"]:
            assert len(data_item["choices_pic"]) > 0
            assert data_item["choices"] is None
            max_choice = max(max_choice, len(data_item["choices_pic"]))
        else:
            assert len(data_item["choices"]) > 0
            assert data_item["choices_pic"] is None
            max_choice = max(max_choice, len(data_item["choices"]))
    print(stem_dataset)
    print("Max choice:", max_choice)
    assert torch.cuda.is_available()
    clip_model, clip_processor = clip.load(args.model_name, device="cuda")
    if args.load_model_dir is not None:
        ckpt = torch.load(args.load_model_dir, map_location="cpu")
        clip_model.load_state_dict(ckpt["model"])
    model = CLIPWrapper(clip_model, clip_processor, args.pic_size, max_choice)
    model.float()
    model.cuda()
    model.eval()
    run_eval(model, stem_dataset, args.eval_split, args.output_dir)


if __name__ == "__main__":
    main()
