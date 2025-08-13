import os
import pickle
import cv2
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import Compose
from data.datasets.utils import LoadImageFromFile
from .preprocess import lane_data_prep

class TusimpleDataset(Dataset):
    def __init__(self, root, split='train', transform=None, version='v1.0'):
        self.root = root
        self.split = split
        self.transform = transform
        self.version = version
        self.ann_file = os.path.join(
            self.root, f'tusimple_infos_{self.split}_{self.version}.pkl'
        )

        # annotation 파일이 없으면 생성
        if not os.path.exists(self.ann_file):
            lane_data_prep(
                root_path=self.root,
                info_prefix='tusimple',
                version=self.version,
                num_workers=32
            )

        self.data_infos = self.load_annotations(self.ann_file)
        self.pipeline = Compose([LoadImageFromFile()])
        self.test_mode = self.split == 'test'
        self.metadata, self.version = None, None

    def load_annotations(self, ann_file):
        with open(ann_file, 'rb') as file:
            data = pickle.load(file)
        self.metadata = data['metadata']
        self.version = self.metadata['version']
        return data['infos']

    def get_data_info(self, index):
        info = self.data_infos[index]
        input_dict = dict(
            lane_points=info['lane_points'],
            img_filename=os.path.join(self.root, info['img_path']),
            img_mask_path=os.path.join(self.root, info['img_mask']),
        )
        return input_dict

    def __getitem__(self, index):
        input_dict = self.get_data_info(index)

        # 이미지 읽기
        img_path = input_dict['img_filename']
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")

        # 이미지 채널 통일
        if img.ndim == 2:  # 흑백 -> 3채널
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:  # RGBA -> BGR
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        input_dict['img'] = img

        # mask 읽기
        mask_path = input_dict.get('img_mask_path', None)
        mask = None
        if mask_path is not None and os.path.exists(mask_path):
            if mask_path.endswith('.npz'):
                mask_data = np.load(mask_path)
                key = list(mask_data.keys())[0]
                mask = mask_data[key]
                if mask.ndim == 3:
                    mask = mask[:, :, 0]
            else:
                mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
                if mask is None:
                    mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
                elif mask.ndim == 2:
                    pass
                elif mask.shape[2] == 4:  # RGBA -> GRAY
                    mask = cv2.cvtColor(mask, cv2.COLOR_BGRA2GRAY)
                else:  # BGR -> GRAY
                    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        if mask is None:
            mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)

        # mask를 3채널로 확장
        mask = np.stack([mask]*3, axis=-1)
        input_dict['img_mask'] = mask

        # filename 생성
        input_dict['filename'] = ','.join(img_path.split('/')[-4:])

        # transform 적용
        if self.transform is not None:
            input_dict = self.transform(input_dict)

        return input_dict

    def __len__(self):
        return len(self.data_infos)

