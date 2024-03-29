# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
COCO dataset which returns image_id for evaluation.

Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""
from pathlib import Path

import torch
import torch.utils.data
import torchvision
from pycocotools import mask as coco_mask

#++++++++++++++++++
from PIL import Image
import os
#++++++++++++++++++

import datasets.transforms as T

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class DualModalityCocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder_visible, img_folder_lwir, ann_file, transforms=None, return_masks=True):
        # Initialize using the visible images and annotations
        super().__init__(img_folder_visible, ann_file)
        self.img_folder_lwir = img_folder_lwir
        self.transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks)
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#-----------------------------------------------------------------

# class CocoDetection(torchvision.datasets.CocoDetection):
#     def __init__(self, img_folder, ann_file, transforms, return_masks):
#         super(CocoDetection, self).__init__(img_folder, ann_file)
#         self._transforms = transforms
#         self.prepare = ConvertCocoPolysToMask(return_masks)

#-----------------------------------------------------------------

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def __getitem__(self, idx):
        # Load visible image and target
        img_visible, target = super().__getitem__(idx)

        # Load corresponding LWIR image
        img_id = self.ids[idx]
        lwir_filename = self.coco.loadImgs(img_id)[0]['file_name']
        lwir_path = os.path.join(self.img_folder_lwir, lwir_filename)
        img_lwir = Image.open(lwir_path).convert("RGB")

        # Apply transforms here if necessary
        if self.transforms is not None:
            img_visible, target = self.transforms(img_visible, target)
            img_lwir, _ = self.transforms(img_lwir, target)  # Assuming same transforms; adjust if needed

        # Return both images and the target
        return img_visible, img_lwir, target

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++    
    # def __getitem__(self, idx):
    #     img, target = super(CocoDetection, self).__getitem__(idx)
    #     image_id = self.ids[idx]
    #     target = {'image_id': image_id, 'annotations': target}
    #     img, target = self.prepare(img, target)
    #     if self._transforms is not None:
    #         img, target = self._transforms(img, target)
    #     return img, target


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        return image, target


def make_coco_transforms(image_set):

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.RandomResize(scales, max_size=1333),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales, max_size=1333),
                ])
            ),
            normalize,
        ])

    if image_set == 'val':
        return T.Compose([
            T.RandomResize([800], max_size=1333),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def build(image_set, args):
    root_visible = Path(args.data_path_visible)
    root_lwir = Path(args.data_path_lwir)
    assert root_visible.exists(), f'provided visible path {root_visible} does not exist'
    assert root_lwir.exists(), f'provided LWIR path {root_lwir} does not exist'

    PATHS = {
        "train": ("train_set", "annotations/instances_train.json"),
        "val": ("val_set", "annotations/instances_val.json"),
    }

    set_type, ann_file = PATHS[image_set]
    img_folder_visible = root_visible / set_type
    img_folder_lwir = root_lwir / set_type
    ann_file_path = root_visible / ann_file  # Assuming annotations are in the visible directory

    dataset = DualModalityCocoDetection(
        img_folder_visible=img_folder_visible,
        img_folder_lwir=img_folder_lwir,
        ann_file=ann_file_path,
        transforms=make_coco_transforms(image_set),
        return_masks=args.masks
    )
    return dataset

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# def build(image_set, args):
#     root = Path(args.coco_path)
#     assert root.exists(), f'provided COCO path {root} does not exist'
#     mode = 'instances'
#     PATHS = {
#         "train": (root / "train2017", root / "annotations" / f'{mode}_train2017.json'),
#         "val": (root / "val2017", root / "annotations" / f'{mode}_val2017.json'),
#     }

#     img_folder, ann_file = PATHS[image_set]
#     dataset = CocoDetection(img_folder, ann_file, transforms=make_coco_transforms(image_set), return_masks=args.masks)
#     return dataset
