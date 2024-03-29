from torch.utils.data import Dataset
import os
import json
from PIL import Image
import torch
import torchvision.transforms.functional as F
from torchvision.transforms import Resize, Compose, ToTensor, Normalize
from util.misc import NestedTensor
import random

class JointTransform:
    def __init__(self, resize=None, flip_prob=0.5):
        # If resize is not None, apply resizing
        self.resize = Resize(resize) if resize else None
        self.flip_prob = flip_prob
        self.normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __call__(self, rgb_image, ir_image, target=None):
        if self.resize:
            rgb_image = self.resize(rgb_image)
            ir_image = self.resize(ir_image)
        
        # Random Horizontal Flip
        if random.random() < self.flip_prob:
            rgb_image = F.hflip(rgb_image)
            ir_image = F.hflip(ir_image)
            if target is not None:
                target = self.flip_boxes(target, rgb_image.width)

        rgb_tensor = ToTensor()(rgb_image)
        ir_tensor = ToTensor()(ir_image)

        # Normalize
        rgb_tensor = self.normalize(rgb_tensor)
        ir_tensor = self.normalize(ir_tensor)

        return rgb_tensor, ir_tensor, target

    def flip_boxes(self, target, image_width):
        if "boxes" in target:
            boxes = target["boxes"]
            boxes = boxes.clone()
            boxes[:, [0, 2]] = image_width - boxes[:, [2, 0]] - 1
            target["boxes"] = boxes
        return target

class DualModalityDataset(Dataset):
    def __init__(self, root_dir_visible, root_dir_lwir, set_type='train_set', joint_transform=None, return_masks=True):
        self.root_dir_visible = root_dir_visible
        self.root_dir_lwir = root_dir_lwir
        self.set_type = set_type
        self.joint_transform = joint_transform
        self.return_masks = return_masks
        
        annotations_path = os.path.join(self.root_dir_visible, set_type, 'annotations.json')
        with open(annotations_path, 'r') as f:
            self.annotations = json.load(f)

        self.id_to_annotations = {anno['image_id']: [] for anno in self.annotations['annotations']}
        for anno in self.annotations['annotations']:
            self.id_to_annotations[anno['image_id']].append(anno)

    def __len__(self):
        return len(self.annotations['images'])

    def __getitem__(self, idx):
        img_info = self.annotations['images'][idx]
        img_id = img_info['id']
        img_annotations = self.id_to_annotations.get(img_id, [])

        rgb_img_path = os.path.join(self.root_dir_visible, self.set_type, img_info['file_name'])
        ir_img_path = os.path.join(self.root_dir_lwir, self.set_type, img_info['file_name'])
        rgb_img = Image.open(rgb_img_path).convert('RGB')
        ir_img = Image.open(ir_img_path).convert('RGB')

        targets = {"boxes": torch.as_tensor([anno['bbox'] for anno in img_annotations], dtype=torch.float32),
                   "labels": torch.as_tensor([anno['category_id'] for anno in img_annotations], dtype=torch.int64),
                   "iscrowd": torch.zeros((len(img_annotations),), dtype=torch.int64)}

        if self.joint_transform:
            rgb_img, ir_img, targets = self.joint_transform(rgb_img, ir_img, targets)

        # Padding and mask generation
        max_size = tuple(max(s) for s in zip(rgb_img.shape, ir_img.shape))
        rgb_padded, rgb_mask = self.pad_and_create_mask(rgb_img, max_size)
        ir_padded, ir_mask = self.pad_and_create_mask(ir_img, max_size)

        if self.return_masks:
            return NestedTensor(rgb_padded, rgb_mask), NestedTensor(ir_padded, ir_mask), targets
        else:
            return rgb_padded, ir_padded, targets

    def pad_and_create_mask(self, img_tensor, max_size):
        """
        Pad the image tensor to max_size and create a mask indicating the padded area.
        """
        pad_height = max_size[1] - img_tensor.shape[1]
        pad_width = max_size[2] - img_tensor.shape[2]
        padded_img = F.pad(img_tensor, (0, pad_width, 0, pad_height), value=0)
        mask = torch.zeros((max_size[1], max_size[2]), dtype=torch.bool)
        mask[:img_tensor.shape[1], :img_tensor.shape[2]] = 1  # Non-padded areas are True
        return padded_img, mask
