from packaging import version
from PIL import Image
from torchvision import transforms
import os
import PIL
from torch.utils.data import Dataset
import torchvision
import numpy as np
import torch
import random
import copy
import cv2
import pandas as pd
import glob

from utils.utils_image import uint2single, single2uint, img2tensor, tensor2uint
from torchvision.transforms import CenterCrop
from utils.utils_image import imfrombytes, img2tensor, padding, padding_DP, imfrombytesDP, uint2single, single2uint, tensor2img
from data.transforms import augment, paired_random_crop, paired_random_crop_DP, random_augmentation

from data.utils_data import (paired_paths_from_folder,
                             paired_DP_paths_from_folder,
                             paired_paths_from_lmdb,
                             paired_paths_from_meta_info_file,
                             tri_paths_from_folder)

imagenet_templates_small = [
    "a photo of a {}",
    "a rendering of a {}",
    "a cropped photo of the {}",
    "the photo of a {}",
    "a photo of a clean {}",
    "a photo of a dirty {}",
    "a dark photo of the {}",
    "a photo of my {}",
    "a photo of the cool {}",
    "a close-up photo of a {}",
    "a bright photo of the {}",
    "a cropped photo of a {}",
    "a photo of the {}",
    "a good photo of the {}",
    "a photo of one {}",
    "a close-up photo of the {}",
    "a rendition of the {}",
    "a photo of the clean {}",
    "a rendition of a {}",
    "a photo of a nice {}",
    "a good photo of a {}",
    "a photo of the nice {}",
    "a photo of the small {}",
    "a photo of the weird {}",
    "a photo of the large {}",
    "a photo of a cool {}",
    "a photo of a small {}",
]


if version.parse(version.parse(PIL.__version__).base_version) >= version.parse("9.1.0"):
    PIL_INTERPOLATION = {
        "linear": PIL.Image.Resampling.BILINEAR,
        "bilinear": PIL.Image.Resampling.BILINEAR,
        "bicubic": PIL.Image.Resampling.BICUBIC,
        "lanczos": PIL.Image.Resampling.LANCZOS,
        "nearest": PIL.Image.Resampling.NEAREST,
    }
else:
    PIL_INTERPOLATION = {
        "linear": PIL.Image.LINEAR,
        "bilinear": PIL.Image.BILINEAR,
        "bicubic": PIL.Image.BICUBIC,
        "lanczos": PIL.Image.LANCZOS,
        "nearest": PIL.Image.NEAREST,
    }


def is_image(file):
    return 'jpg' in file.lower()  or 'png' in file.lower()  or 'jpeg' in file.lower()


###############################
# newly added
###############################

# used to train image-to-text mapper
class UnpairedLQHQDataset(Dataset):
    def __init__(self,
                 csv_path,  # <- 修改1：增加csv_path参数
                 tokenizer,
                 size=512,
                 interpolation="bicubic",
                 placeholder_token="*",
                 template="a photo of a {}"):
        super(UnpairedLQHQDataset, self).__init__()

        # 修改2：从csv加载数据路径和标题
        self.df = pd.read_csv(csv_path, sep='\t')  # 假设csv用制表符分隔

        # 定义函数提取退化类型
        def extract_degradation_type(path):
            parts = path.split('/')
            train_index = parts.index('train')  # 找到train的位置
            return parts[train_index + 1]       # train下一级目录是退化类型

        # 添加退化类型列
        self.df['degration_type'] = self.df['filepath'].apply(extract_degradation_type)

        # 按退化类型分组并取前2000条
        self.df = self.df.groupby('degration_type').head(2000).reset_index(drop=True)

        # 生成对应的GT路径
        self.df['GT_path'] = self.df['filepath'].str.replace('/LQ/', '/GT/')

        self.image_paths = self.df['filepath'].tolist()
        self.GT_paths = self.df['GT_path'].tolist() # 添加GT路径
        # Update paths based on new base directory
        self.image_paths = [p.replace("/workspace/datasets/SD_Rest", "/data/coding") for p in self.image_paths]
        self.GT_paths = [p.replace("/workspace/datasets/SD_Rest", "/data/coding") for p in self.GT_paths]
        self.titles = [x.split(":")[0].strip() for x in self.df['title']]  # 提取冒号前的描述

        self.title_templates = [x.split(":")[0].strip() + " {}" for x in self.df['title']]

        # 修改3：调整类成员变量
        self.num_images = len(self.image_paths)
        self._length = self.num_images
        self.tokenizer = tokenizer
        self.size = size
        self.interpolation = {
            "linear": PIL_INTERPOLATION["linear"],
            "bilinear": PIL_INTERPOLATION['bilinear'],
            "bicubic": PIL_INTERPOLATION["bicubic"],
            "lanczos": PIL_INTERPOLATION["lanczos"]
        }[interpolation]

        
        # 废弃原来的路径收集逻辑
        # self.dataroot_list = dataroot_list
        # self.image_paths = []
        # for dataroot in self.dataroot_list:
        #     self.image_paths.extend(sorted(glob.glob(os.path.join(dataroot, "*"))))
        self.placeholder_token = placeholder_token
        self.template = template
        self.patch_size = size
        """
        self.tokenizer = tokenizer
        self.size = size

        self.num_images = len(self.image_paths)
        self._length = self.num_images

        self.interpolation = {
            "linear": PIL_INTERPOLATION["linear"],
            "bilinear": PIL_INTERPOLATION['bilinear'],
            "bicubic": PIL_INTERPOLATION["bicubic"],
            "lanczos": PIL_INTERPOLATION["lanczos"]
        }[interpolation]

        """

    def __len__(self):
        return self._length

    def get_tensor_clip(self, normalize=True, toTensor=True):
        transform_list = []
        if toTensor:
            transform_list += [torchvision.transforms.ToTensor()]

        if normalize:
            transform_list += [torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                                                (0.26862954, 0.26130258, 0.27577711))]

        return torchvision.transforms.Compose(transform_list)

    def process(self, image):

        img = cv2.resize(image, (self.size, self.size), interpolation=cv2.INTER_CUBIC)

        img = np.array(img).astype(np.float32)
        img = img / 127.5 - 1.0
        return torch.from_numpy(img).permute(2, 0, 1)

    def __getitem__(self, i):

        ###############################################################
        example = {}

        # 修改4：使用csv中的真实文本
        
        # raw_text = self.titles[i % self.num_images]  # 直接使用标题中的描述
        # example["text"] = raw_text
        """
        # 修改5：简化tokenize流程（不需要占位符处理）
        example["input_ids"] = self.tokenizer(
            raw_text,  # 直接使用原始文本
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids[0]
        """
        placeholder_string = self.placeholder_token
        text = self.template.format(placeholder_string)
        # text = self.title_templates[i % self.num_images].format(placeholder_string)
        example["text"] = text

        placeholder_index = 0
        words = text.strip().split(' ')
        for idx, word in enumerate(words):
            if word == placeholder_string:
                placeholder_index = idx + 1

        example["index"] = torch.tensor(placeholder_index)
        example["input_ids"] = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt"
        ).input_ids[0]

        # 修改6：保留原有图像处理逻辑
        self.current_path = self.image_paths[i % self.num_images]
        image_name = self.current_path.split('/')[-1].split(".")[0]

        self.current_path_gt = self.GT_paths[i % self.num_images]
        image_name_gt = self.current_path_gt.split('/')[-1].split(".")[0]

        try:
            image = Image.open(self.current_path)
        
            if not image.mode == "RGB":
                image = image.convert("RGB")


            image_gt = Image.open(self.current_path_gt)
        
            if not image_gt.mode == "RGB":
                image_gt = image_gt.convert("RGB")

            H, W = image.size
            # 定义通用预处理（排除 RandomCrop）
            process = []

            # 需要裁剪时
            if H >= self.patch_size and W >= self.patch_size:
                # 获取随机裁剪参数
                i, j, h, w = transforms.RandomCrop.get_params(
                    image, output_size=(self.patch_size, self.patch_size)
                )
                
                # 对两个图像应用相同的裁剪坐标
                process += transforms.Lambda(lambda x: transforms.functional.crop(x, i, j, h, w)),
                # process += transforms.Crop(image, i, j, h, w)
                
                # 统一缩放到最终尺寸
                # process += torchvision.transforms.Resize(self.size, interpolation=self.interpolation),
            else:
                # 小图直接中心裁剪+缩放
                process += torchvision.transforms.Resize(self.size, interpolation=self.interpolation),
                process += torchvision.transforms.CenterCrop(self.size),
            
            process += torchvision.transforms.ToTensor(),
            process = torchvision.transforms.Compose(process)
            # 转换为 Tensor
            torch_image = process(image)
            torch_image_gt = process(image_gt)

            """process = []
            if H < self.patch_size or W < self.patch_size:
                process += torchvision.transforms.Resize(self.size, interpolation=self.interpolation),
                process += torchvision.transforms.CenterCrop(self.size),
            else:
                process += torchvision.transforms.RandomCrop(self.size),
                process += torchvision.transforms.Resize(self.size, interpolation=self.interpolation),
                
            process += torchvision.transforms.ToTensor(),
            process = torchvision.transforms.Compose(process)

            torch_image = process(image)
            torch_image_gt = process(image_gt)"""

            example["pixel_values"] = torch_image
            example["pixel_values_vae"] = torchvision.transforms.Normalize(
                mean=[0.5],
                std=[0.5],
            )(torch_image)

            example["pixel_values_vae_gt"] = torchvision.transforms.Normalize(
                mean=[0.5],
                std=[0.5],
            )(torch_image_gt)


            example["pixel_values_clip"] = torchvision.transforms.Compose( # the clip process input range should be [0, 1]
                [torchvision.transforms.Resize((224, 224), interpolation=self.interpolation),
                    torchvision.transforms.Normalize(
                        mean=[0.48145466, 0.4578275, 0.40821073],
                        std=[0.26862954, 0.26130258, 0.27577711]
                    )
                    ]
            )(torch_image)

            """
            # PIL Image
            H, W = image.size
            if H < self.patch_size or W < self.patch_size:
                croper = torchvision.transforms.CenterCrop(H if H < W else W)
                image = croper(image)
                image_np = np.array(image)
            else:
                image_np = np.array(image)
                rnd_h_H = random.randint(0, max(0, H - self.patch_size))
                rnd_w_H = random.randint(0, max(0, W - self.patch_size))
                image_np = image_np[rnd_w_H : rnd_w_H + self.patch_size, rnd_h_H: rnd_h_H + self.patch_size,:]

            image_np = uint2single(image_np)

            example["pixel_values"] = self.process(image_np)

            ref_image_tensor = Image.fromarray(image_np.astype('uint8')).resize((224, 224), resample=self.interpolation)

            example["pixel_values_clip"] = self.get_tensor_clip()(ref_image_tensor)
            """

            example["image_name"] = image_name

        except Exception as e:

            example["pixel_values"] = torch.zeros((3, 512, 512))
            example["pixel_values_clip"] = torch.zeros((3, 224, 224))
            example["pixel_values_clip_save"] = torch.zeros((3, 512, 512))

            example["image_name"] = image_name

            print("Bad Image Path", self.current_path)

        return example


# used to train text restoration mapper
class PairedLQHQDataset(Dataset):
    def __init__(self,
                 task_list,
                 dataroot_list,
                 tokenizer,
                 size=512,
                 interpolation="bicubic",
                 placeholder_token="*",
                 template="a photo of a {}"):
        super(PairedLQHQDataset, self).__init__()

        self.dataroot_list = dataroot_list
        self.task_list = task_list
        self.tokenizer = tokenizer
        self.size = size
        self.placeholder_token = placeholder_token
        self.patch_size = size

        assert len(self.dataroot_list) == len(self.task_list), "task and dataroot should be aligned"

        self.tasks = list(set(self.task_list))
        self.num_task = len(list(set(self.task_list)))

        self.image_paths_lq = {}
        self.image_paths_hq = {}

        for task_ in self.tasks:
            self.image_paths_lq[task_] = []
            self.image_paths_hq[task_] = []

        for task_, dataroot_ in zip(self.task_list, self.dataroot_list):
            if task_ == "denoise":
                self.image_paths_hq[task_].extend(sorted(glob.glob(os.path.join(dataroot_, "*"))))
            elif task_ == "deblur":
                self.image_paths_lq[task_].extend(sorted(glob.glob(os.path.join(dataroot_, "lq", "*"))))
                self.image_paths_hq[task_].extend(sorted(glob.glob(os.path.join(dataroot_, "hq", "*"))))
            elif task_ == "derain":
                self.image_paths_lq[task_].extend(sorted(glob.glob(os.path.join(dataroot_, "lq", "*"))))
                self.image_paths_hq[task_].extend(sorted(glob.glob(os.path.join(dataroot_, "hq", "*"))))
            elif task_ == "dehaze":
                meta_info_file = os.path.join(dataroot_, "meta_info.txt")
                with open(meta_info_file) as f:
                    contents = f.readlines()
                    haze_names_indoor = [i.strip() for i in contents]
                    gt_names_indoor = [i.split('_')[0] for i in haze_names_indoor]  # haze_names#
                self.image_paths_lq[task_].extend([os.path.join(dataroot_, i) for i in haze_names_indoor])
                self.image_paths_hq[task_].extend([os.path.join(dataroot_, i) for i in gt_names_indoor])
            else:
                raise NotImplementedError

        # self.image_paths = []
        # self.image_lq_paths = []
        # self.image_hq_paths = []

        # including both high-quality images and degradaed images
        # for dataroot in self.dataroot_list:
        #     self.image_lq_paths.extend(sorted(glob.glob(os.path.join(dataroot, "lq", "*"))))
        #     self.image_hq_paths.extend(sorted(glob.glob(os.path.join(dataroot, "hq", "*"))))

        # denoise, derain, dehaze, deblur
        # dataroot/denoise/dataset_name/
        # dataroot/derain
        # dataroot/dehaze
        # dataroot/deblur

        # self.denoise_hq_image_paths = []
        #
        # self.derain_lq_image_paths = []
        # self.derain_hq_image_paths = []
        #
        # self.deblur_lq_image_paths = []
        # self.deblur_hq_image_paths = []
        #
        # self.dehaze_lq_image_paths = []
        # self.dehaze_hq_image_paths = []

        # folder_roots = sorted(glob.glob(os.path.join(self.dataroot, "*")))
        # for folder_root in folder_roots:
        #     self.image_lq_paths.extend(sorted(glob.glob(os.path.join(folder_root, "lq", "*"))))
        #     self.image_hq_paths.extend(sorted(glob.glob(os.path.join(folder_root, "hq", "*"))))

        # default
        # denoise_root = os.path.join(self.dataroot, "denoise")
        #
        # denoise_folders = sorted(glob.glob(os.path.join(denoise_root, "*")))
        # for denoise_folder in denoise_folders:
        #     self.denoise_hq_image_paths.extend(sorted(glob.glob(os.path.join(denoise_folder, "*"))))
        #
        # derain_root = os.path.join(self.dataroot, "derain")
        # derain_folders = sorted(glob.glob(os.path.join(derain_root, "*")))
        # for derain_folder in derain_folders:
        #     self.derain_hq_image_paths.extend(sorted(glob.glob(os.path.join(derain_folder, "hq", "*"))))
        #     self.derain_lq_image_paths.extend(sorted(glob.glob(os.path.join(derain_folder, "lq", "*"))))
        #
        # deblur_root = os.path.join(self.dataroot, "deblur")
        # deblur_folders = sorted(glob.glob(os.path.join(deblur_root, "*")))
        # for deblur_folder in deblur_folders:
        #     self.deblur_lq_image_paths.extend(sorted(glob.glob(os.path.join(deblur_folder, "lq", "*"))))
        #     self.deblur_hq_image_paths.extend(sorted(glob.glob(os.path.join(deblur_folder, "hq", "*"))))
        #
        # dehaze_root = os.path.join(self.dataroot, "dehaze")
        # dehaze_indoor_meta_info_file = os.path.join(self.dataroot, "dehaze", "indoor", "meta_info.txt")
        # dehaze_outdoor_meta_info_file = os.path.join(self.dataroot, "dehaze", "outdoor", "meta_info.txt")
        # with open(dehaze_indoor_meta_info_file) as f:
        #     contents = f.readlines()
        #     haze_names_indoor = [i.strip() for i in contents]
        #     gt_names_indoor = [i.split('_')[0] for i in haze_names_indoor]  # haze_names#
        #
        # dehaze_indoor_lq_paths = [os.path.join(dehaze_root, "indoor", "lq", i) for i in haze_names_indoor]
        # dehaze_indoor_hq_paths = [os.path.join(dehaze_root, "indoor", "hq", i) for i in gt_names_indoor]
        #
        # self.dehaze_lq_image_paths.extend(dehaze_indoor_lq_paths)
        # self.dehaze_hq_image_paths.extend(dehaze_indoor_hq_paths)
        #
        # with open(dehaze_outdoor_meta_info_file) as f:
        #     contents = f.readlines()
        #     haze_names_outdoor = [i.strip() for i in contents]
        #     gt_names_outdoor = [i.split('_')[0] for i in haze_names_outdoor]  # haze_names#
        #
        # dehaze_outdoor_lq_paths = [os.path.join(dehaze_root, "outdoor", "lq", i) for i in haze_names_outdoor]
        # dehaze_outdoor_hq_paths = [os.path.join(dehaze_root, "outdoor", "hq", i) for i in gt_names_outdoor]
        #
        # self.dehaze_lq_image_paths.extend(dehaze_outdoor_lq_paths)
        # self.dehaze_hq_image_paths.extend(dehaze_outdoor_hq_paths)
        #
        # self.num_images = len(self.denoise_hq_image_paths) + len(self.derain_hq_image_paths) + \
        #                   len(self.deblur_hq_image_paths) + len(self.dehaze_hq_image_paths)

        self.num_images = 0
        for task_ in self.tasks:
            self.num_images = self.num_images + len(self.image_paths_hq[task_])

        self._length = self.num_images

        self.interpolation = {
            "linear": PIL_INTERPOLATION["linear"],
            "bilinear": PIL_INTERPOLATION['bilinear'],
            "bicubic": PIL_INTERPOLATION["bicubic"],
            "lanczos": PIL_INTERPOLATION["lanczos"]
        }[interpolation]

        self.template = template
        self.bad_image_list = []

    def __len__(self):
        return self._length

    def get_tensor_clip(self, normalize=True, toTensor=True):
        transform_list = []
        if toTensor:
            transform_list += [torchvision.transforms.ToTensor()]

        if normalize:
            transform_list += [torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                                                (0.26862954, 0.26130258, 0.27577711))]

        return torchvision.transforms.Compose(transform_list)

    def process(self, image):

        img = cv2.resize(image, (self.size, self.size), interpolation=cv2.INTER_CUBIC)

        img = np.array(img).astype(np.float32)
        img = img / 127.5 - 1.0
        return torch.from_numpy(img).permute(2, 0, 1)

    def __getitem__(self, i):

        ############################################################
        example = {}

        placeholder_string = self.placeholder_token
        text = self.template.format(placeholder_string)
        example["text"] = text

        placeholder_index = 0
        words = text.strip().split(' ')
        for idx, word in enumerate(words):
            if word == placeholder_string:
                placeholder_index = idx + 1

        example["index"] = torch.tensor(placeholder_index)
        example["input_ids"] = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids[0]

        #############################################################

        # task 0 denoise
        # task 1 deblur
        # task 2 derain
        # task 3 dehaze
        choice = random.choice([i in range(len(self.tasks))])
        current_task = self.tasks[choice]
        # denoise
        if current_task == "denoise":
            self.current_hq_path = self.image_paths_hq[current_task][i % len(self.image_paths_hq[current_task])]
            self.current_lq_path = None
            hq_image = Image.open(self.current_hq_path)
            lq_image = hq_image.copy()

        # deblur, derain, dehaze
        elif current_task == "deblur" or current_task == "derain" or current_task == "dehaze":
            self.current_hq_path = self.image_paths_hq[current_task][i % len(self.image_paths_hq[current_task])]
            self.current_lq_path = self.image_paths_lq[current_task][i % len(self.image_paths_lq[current_task])]
            hq_image = Image.open(self.current_hq_path)
            lq_image = Image.open(self.current_lq_path)
        else:
            raise NotImplementedError

        image_name = self.current_hq_path.split("/")[-1].split(".")[0]

        try:
            if not lq_image.mode == "RGB":
                lq_image = lq_image.convert("RGB")

            if not hq_image.mode == "RGB":
                hq_image = hq_image.convert("RGB")

            H, W = hq_image.size

            # center crop or random crop
            if H < self.patch_size or W < self.patch_size:

                croper = torchvision.transforms.CenterCrop(H if H < W else W)
                lq_image = croper(lq_image)
                hq_image = croper(hq_image)

                lq_image_np = np.array(lq_image)
                hq_image_np = np.array(hq_image)

            else:
                lq_image_np = np.array(lq_image)
                hq_image_np = np.array(hq_image)
                rnd_h_H = random.randint(0, max(0, H - self.patch_size))
                rnd_w_H = random.randint(0, max(0, W - self.patch_size))
                lq_image_np = lq_image_np[rnd_h_H: rnd_h_H + self.patch_size, rnd_w_H:rnd_w_H + self.patch_size, :]
                hq_image_np = hq_image_np[rnd_h_H: rnd_h_H + self.patch_size, rnd_w_H:rnd_w_H + self.patch_size, :]

            # without adding synthesized degradation
            lq_image_np = uint2single(lq_image_np)
            hq_image_np = uint2single(hq_image_np)

            if current_task == "denoise":
                # todo add random noise [0, 50]
                lq_image_tensor = img2tensor(lq_image_np, bgr2rgb=False, float32=True)
                sigma_value = random.uniform(0.0, 50.0)
                noise_level = torch.FloatTensor([sigma_value]) / 255.
                noise = torch.randn(lq_image_tensor.size()).mul_(noise_level).float()
                lq_image_tensor.add_(noise)

                lq_image_np = tensor2uint(lq_image_tensor)
                hq_image_np = single2uint(hq_image_np)

            else:
                lq_image_np = single2uint(lq_image_np)
                hq_image_np = single2uint(hq_image_np)


            example["pixel_values"] = self.process(hq_image_np)

            ref_image_tensor = Image.fromarray(lq_image_np.astype('uint8')).resize((224, 224), resample=self.interpolation)
            ref_image_tensor_save = Image.fromarray(lq_image_np.astype('uint8')).resize((512, 512), resample=self.interpolation)

            example["pixel_values_clip"] = self.get_tensor_clip()(ref_image_tensor)
            example["pixel_values_clip_save"] = self.get_tensor_clip()(ref_image_tensor_save)

            example["image_name"] = image_name

        except Exception as e:

            example["pixel_values"] = torch.zeros((3, 512, 512))
            example["pixel_values_clip"] = torch.zeros((3, 224, 224))
            example["pixel_values_clip_save"] = torch.zeros((3, 512, 512))

            example["image_name"] = image_name

            print("Bad Image Path", self.current_hq_path)

        return example


# used to generate guidance images
class ReferenceGenerationDataset(Dataset):
    def __init__(self,
                 dataroot,
                 range,
                 tokenizer,
                 size=512,
                 interpolation="bicubic",
                 placeholder_token="*",
                 template="a photo of a {}"):
        super(ReferenceGenerationDataset, self).__init__()

        self.dataroot = dataroot
        self.tokenizer = tokenizer
        self.size = size
        self.placeholder_token = placeholder_token
        self.patch_size = size

        self.image_paths = []

        self.image_paths.extend(sorted(glob.glob(os.path.join(self.dataroot, "*"))))

        if range is not None:
            self.image_paths = self.image_paths[range[0]:range[1]]

        self.num_images = len(self.image_paths)
        self._length = self.num_images

        self.interpolation = {
            "linear": PIL_INTERPOLATION["linear"],
            "bilinear": PIL_INTERPOLATION['bilinear'],
            "bicubic": PIL_INTERPOLATION["bicubic"],
            "lanczos": PIL_INTERPOLATION["lanczos"]
        }[interpolation]

        self.template = template
        self.bad_image_list = []

    def __len__(self):
        return self._length

    def get_tensor_clip(self, normalize=True, toTensor=True):
        transform_list = []
        if toTensor:
            transform_list += [torchvision.transforms.ToTensor()]

        if normalize:
            transform_list += [torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                                                (0.26862954, 0.26130258, 0.27577711))]

        return torchvision.transforms.Compose(transform_list)

    def process(self, image):

        img = cv2.resize(image, (self.size, self.size), interpolation=cv2.INTER_CUBIC)

        img = np.array(img).astype(np.float32)
        img = img / 127.5 - 1.0
        return torch.from_numpy(img).permute(2, 0, 1)

    def __getitem__(self, i):

        ###########################################################
        example = {}

        placeholder_string = self.placeholder_token
        text = self.template.format(placeholder_string)
        example["text"] = text

        placeholder_index = 0
        words = text.strip().split(' ')
        for idx, word in enumerate(words):
            if word == placeholder_string:
                placeholder_index = idx + 1

        example["index"] = torch.tensor(placeholder_index)
        example["input_ids"] = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids[0]

        #############################################################

        self.image_path = self.image_paths[i % self.num_images]

        image = Image.open(self.image_path)

        image_name = self.image_path.split('/')[-1].split(".")[0]

        try:
            if not image.mode == "RGB":
                image = image.convert("RGB")

            H, W = image.size

            # center crop
            if H < W:
                croper = CenterCrop(H)
            else:
                croper = CenterCrop(W)

            image = croper(image)
            image_np = np.array(image)

            # without adding synthesized degradation
            image_np = uint2single(image_np)
            image_np = single2uint(image_np)

            example["pixel_values"] = self.process(image_np)

            ref_image_tensor = Image.fromarray(image_np.astype('uint8')).resize((224, 224), resample=self.interpolation)
            ref_image_tensor_save = Image.fromarray(image_np.astype('uint8')).resize((512, 512), resample=self.interpolation)

            example["pixel_values_clip"] = self.get_tensor_clip()(ref_image_tensor)
            example["pixel_values_clip_save"] = self.get_tensor_clip()(ref_image_tensor_save)

            example["image_name"] = image_name

        except Exception as e:

            example["pixel_values"] = torch.zeros((3, 512, 512))
            example["pixel_values_clip"] = torch.zeros((3, 224, 224))
            example["pixel_values_clip_save"] = torch.zeros((3, 512, 512))

            example["image_name"] = image_name

            print("Bad Image Path", self.image_path)

        return example



