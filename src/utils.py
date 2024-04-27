import io
import torch
import json
import os
import clip
from PIL import Image
_BOSToken = clip.tokenize("")[0,0:1]
_EOSToken = clip.tokenize("")[0,1:2]
MAX_TEXT_LEN = 77


def get_text_info(data_item):
    text = data_item['problem']
    textToken = clip.tokenize(text, truncate=True)
    choices = data_item['choices']
    is_text_choice = not data_item["pic_choice"]
    if not is_text_choice:
        return textToken

    textTokenEOS = textToken.argmax(dim=-1).item()
    textTokenPure = textToken[0, 1:textTokenEOS]
    textTokenPureLen = textTokenEOS - 1
    textConcatChoiceList = []
    assert len(choices) > 1
    for choice in choices:
        choiceToken = clip.tokenize(choice, truncate=True)
        choiceEos = choiceToken.argmax(dim=-1).item()
        choiceTokenPure = choiceToken[0, 1:choiceEos]
        choiceTokenPureLen = choiceEos - 1
        truncatedTextLen = min(MAX_TEXT_LEN - choiceTokenPureLen - 2, textTokenPureLen)
        paddings = torch.tensor([0]* (MAX_TEXT_LEN - choiceTokenPureLen - 2 - truncatedTextLen), dtype=torch.int32)
        textConcatChoice = torch.cat([_BOSToken, textTokenPure[:truncatedTextLen], choiceTokenPure, _EOSToken, paddings])
        assert textConcatChoice.shape[0] == MAX_TEXT_LEN and len(textConcatChoice.shape) == 1
        textConcatChoiceList.append(textConcatChoice)
    return torch.stack(textConcatChoiceList)


def get_data_point(data_item: dict):
    answer_id = data_item["answer_idx"]
    is_text_choice = not data_item["pic_choice"]
    text_info = get_text_info(data_item)
    assert len(text_info.shape) == 2 and text_info.shape[1] == MAX_TEXT_LEN

    if data_item["pic_prob"]:
        pic_info = bytes_to_image(data_item["problem_pic"])
    else:
        assert data_item["pic_choice"]
        pic_info = [bytes_to_image(pic) for pic in data_item["choices_pic"]]
    return {
        "text_info": text_info,
        "pic_info": pic_info,
        "answer_id": answer_id,
        "is_text_choice": is_text_choice,
    }


class CLIPWrapper(torch.nn.Module):
    def __init__(self, clip_model, clip_processor, pic_size: int, max_choice: int):
        super().__init__()
        self.model = clip_model
        self.processor = clip_processor
        self.pic_size = pic_size
        self.max_choice = max_choice

    def forward(self, text_input, image_input):
        batch_size = text_input.shape[0] // self.max_choice
        text_features = self.model.encode_text(text_input)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
        text_features = text_features.reshape(batch_size, self.max_choice, -1)

        image_features = self.model.encode_image(image_input)
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        image_features = image_features.reshape(batch_size, self.max_choice, -1)

        logit_scale = self.model.logit_scale.exp()
        logits_per_pair = logit_scale * torch.sum(
            text_features * image_features, dim=-1
        )
        return logits_per_pair


def bytes_to_image(img_bytes: bytes) -> Image:
    img = Image.open(io.BytesIO(img_bytes))
    return img


def get_real_pic_info(pic_obj, processor):
    middle_after = processor(pic_obj).to("cuda").unsqueeze(0)
    return middle_after


def collate_fn(batch_of_data, max_choice, pic_size, processor):
    batch_text_info, batch_pic_info, batch_pad_mask, batch_answer_id = [], [], [], []
    for data in batch_of_data:
        batch_answer_id.append(data["answer_id"])
        text_info, pic_info = data["text_info"], data["pic_info"]
        if isinstance(pic_info, list):
            pic_info = [get_real_pic_info(pic, processor) for pic in pic_info]
            pic_info = torch.cat(pic_info, dim=0)
        else:
            pic_info = get_real_pic_info(pic_info, processor)
        assert text_info.shape[0] <= max_choice and pic_info.shape[0] <= max_choice
        if text_info.shape[0] == 1:
            text_info = text_info.repeat(pic_info.shape[0], 1)
        else:
            assert pic_info.shape[0] == 1
            pic_info = pic_info.repeat(text_info.shape[0], 1, 1, 1)

        pad_dim = max_choice - text_info.shape[0]
        batch_pad_mask.append(
            torch.tensor([0.0] * text_info.shape[0] + [-1e8] * pad_dim)
        )
        if pad_dim > 0:
            text_pad = torch.zeros((pad_dim, MAX_TEXT_LEN)).type_as(text_info)
            pic_pad = torch.zeros((pad_dim, 3, pic_size, pic_size)).type_as(pic_info)
            text_info = torch.cat([text_info, text_pad], dim=0)
            pic_info = torch.cat([pic_info, pic_pad], dim=0)
        batch_text_info.append(text_info)
        batch_pic_info.append(pic_info)
    batch_text_info = torch.cat(batch_text_info, dim=0)
    batch_pic_info = torch.cat(batch_pic_info, dim=0)
    batch_pad_mask = torch.stack(batch_pad_mask, dim=0)
    batch_answer_id = torch.tensor(batch_answer_id).type(torch.int32)
    return {
        "text_info": batch_text_info,
        "pic_info": batch_pic_info,
        "pad_mask": batch_pad_mask,
        "answer_id": batch_answer_id,
    }
