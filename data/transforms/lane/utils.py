import re
import torch
from torch import Tensor
from typing import Optional, List
from collections import defaultdict
import numpy as np

np_str_obj_array_pattern = re.compile(r'[SaUO]')

def _max_by_axis(the_list: List[List[int]]) -> List[int]:
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes

def nested_tensor_from_tensor_list(tensor_list: List[Tensor]):
    # 모든 텐서 채널 수를 3으로 통일
    processed_list = []
    for img in tensor_list:
        if img.ndim == 3:
            if img.shape[0] == 1:
                img = img.repeat(3, 1, 1)
            elif img.shape[0] == 4:
                img = img[:3, :, :]  # RGBA -> RGB
        processed_list.append(img)

    max_size = _max_by_axis([list(img.shape) for img in processed_list])
    batch_shape = [len(processed_list)] + max_size
    dtype = processed_list[0].dtype
    device = processed_list[0].device
    tensor = torch.zeros(batch_shape, dtype=dtype, device=device)

    for i, img in enumerate(processed_list):
        if img.ndim == 1:
            tensor[i, ..., :img.shape[-1]].copy_(img)
        else:
            tensor[i, ..., :img.shape[-2], :img.shape[-1]].copy_(img)
    return tensor

def collate_fn(data_list):
    result = defaultdict(list)
    for data in data_list:
        for key, value in data.items():
            result[key].append(value)
    return result

def collate_fn_padded(data_list, keys=['img', 'img_mask']):
    result = defaultdict(list)
    for data in data_list:
        for key, value in data.items():
            # numpy 배열이면 tensor(C,H,W)로 변환
            if key in keys and isinstance(value, np.ndarray):
                if value.ndim == 2:
                    value = np.stack([value]*3, axis=2)
                value = torch.from_numpy(value.transpose(2,0,1))
            result[key].append(value)

    for key in result:
        if key not in keys:
            continue
        if not all([isinstance(x, torch.Tensor) for x in result[key]]):
            continue
        result[key] = nested_tensor_from_tensor_list(result[key])
    return result

