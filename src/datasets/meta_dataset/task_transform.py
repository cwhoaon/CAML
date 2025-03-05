import torch
import torchvision.transforms as transforms
from PIL import ImageEnhance
import random
import math
import torchvision.transforms.functional as F

from .utils import Split
from .config import DataConfig


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

# 기존 ImageJitter 활용
jitter_param = dict(Brightness=0.1, Contrast=0.1, Color=0.1)


class ConsistentRandomTransform:
    """
    텐서 입력(C,H,W)에 대해, 배치 단위로 동일한
    random crop/flip/color jitter 등을 적용하는 예시 클래스.
    - PIL 변환 대신 torchvision.transforms.functional의
      Tensor 전용 함수를 사용.
    """

    def __init__(self,
                 image_size: int,
                 jitter_dict: dict,
                 scale=(0.08, 1.0),
                 ratio=(3. / 4., 4. / 3.)):
        self.image_size = image_size
        self.jitter_dict = jitter_dict  # {'Brightness':0.4, 'Contrast':0.4, ...}
        self.scale = scale
        self.ratio = ratio

        # 일관된 파라미터를 저장할 변수들
        self.crop_params = None  # (top, left, height, width)
        self.do_flip = random.random() < 0.5
        self.jitter_factors = []  # [('Brightness', val), ('Contrast', val), ...]

        # 초기화 시에 랜덤 파라미터 한 번 뽑아서 저장
        self._init_jitter_params()

    def _init_jitter_params(self):
        """
        ColorJitter(혹은 ImageJitter) 로직을 모방하여
        brightness / contrast / color 등의 factor를 미리 뽑아둠.
        """
        randtensor = torch.rand(len(self.jitter_dict))
        for i, (k, alpha) in enumerate(self.jitter_dict.items()):
            # [-alpha, alpha] 범위 내 랜덤 → +1.0 → [1-alpha, 1+alpha] 범위
            factor = alpha * (randtensor[i] * 2.0 - 1.0) + 1.0
            self.jitter_factors.append((k, factor))

    def _get_crop_params(self, h, w):
        """
        RandomResizedCrop과 유사하게 (top, left, height, width)를 구함.
        """
        area = h * w
        for _ in range(10):
            target_area = random.uniform(*self.scale) * area
            log_ratio = (math.log(self.ratio[0]), math.log(self.ratio[1]))
            aspect_ratio = math.exp(random.uniform(*log_ratio))

            nw = int(round(math.sqrt(target_area * aspect_ratio)))
            nh = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < nw <= w and 0 < nh <= h:
                top = random.randint(0, h - nh)
                left = random.randint(0, w - nw)
                return (top, left, nh, nw)

        # 위에서 랜덤 파라미터를 구하지 못했다면 center crop fallback
        in_ratio = float(w) / float(h)
        if in_ratio < self.ratio[0]:
            nw = w
            nh = int(round(w / self.ratio[0]))
        elif in_ratio > self.ratio[1]:
            nh = h
            nw = int(round(h * self.ratio[1]))
        else:
            nw = w
            nh = h
        top = (h - nh) // 2
        left = (w - nw) // 2
        return (top, left, nh, nw)

    def _apply_jitter(self, x: torch.Tensor):
        """
        미리 뽑은 색/밝기/대비 파라미터를 텐서에 적용.
        (C,H,W) 텐서여야 하며, 0~1 범위로 가정.
        """
        for (k, factor) in self.jitter_factors:
            if k == 'Brightness':
                x = F.adjust_brightness(x, factor)
            elif k == 'Contrast':
                x = F.adjust_contrast(x, factor)
            elif k == 'Color':
                # Color = Saturation에 해당
                x = F.adjust_saturation(x, factor)
            elif k == 'Sharpness':
                # torchvision 0.10+에서만 Tensor input 지원
                x = F.adjust_sharpness(x, factor)
        return x

    def __call__(self, x: torch.Tensor):
        """
        x.shape == (C, H, W) 텐서 입력을 가정
        (배치 단위로 동일 파라미터를 쓰고 싶다면,
         한 번 생성한 transform 객체에 대해
         배치 내 모든 샘플 x를 순회하며 __call__하면 됨)
        """
        # H, W 구하기
        _, h, w = x.shape

        # RandomResizedCrop
        if self.crop_params is None:
            self.crop_params = self._get_crop_params(h, w)
        top, left, nh, nw = self.crop_params
        # torchvision.transforms.functional.resized_crop 사용
        x = F.resized_crop(
            x, top, left, nh, nw,
            size=(self.image_size, self.image_size)
        )

        # RandomHorizontalFlip
        if self.do_flip:
            x = F.hflip(x)

        # Color Jitter
        x = self._apply_jitter(x)

        # (선택) 정규화
        x = normalize(x)  # 필요하다면 (C,H,W) 형태 유지
        return x


def get_consistent_transform(data_config: DataConfig):
    """
    ConsistentRandomTransform를 이용하여,
    같은 배치(n개 이미지)에서 동일 파라미터로 변환되도록 구성한 예시.
    DataLoader 등에서 '배치 단위로' 이 변환 객체를 생성해 써야 함.
    """
    return ConsistentRandomTransform(
        image_size=data_config.image_size,
        jitter_dict=jitter_param,
        scale=(0.08, 1.0),
        ratio=(3./4., 4./3.)
    )
