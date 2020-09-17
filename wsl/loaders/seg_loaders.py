# This code does not belong to this repo - just copied here for reference on dicom 2D and 3D loaders

from collections import defaultdict
import functools
from pathlib import Path
import traceback
from typing import Any, Callable, Dict, Sequence, Optional, Tuple, Union, Mapping, List
import random

import numpy as np
import pandas as pd
from skimage import transform
from scipy.special import softmax
import torch
from torch.utils import data

from cspine_detect import locations

# type alias
Array = np.ndarray
MaybeFloat = Optional[float]
MaybeInt = Optional[int]
MaybeStr = Optional[str]
Tensor = torch.Tensor

def check_path_exists(p: Path) -> None:
    if not p.exists():
        raise FileNotFoundError(f'Input file {p} does not exist')
        
def check_shape(A: Array, B: Array) -> None:
    assert A.shape == B.shape, f'\nshape mismatch:\t{A.shape} != {B.shape}'
    
class FractureDataset(data.Dataset):
    '''
    data_split - split to use (one of 'train', 'valid', or 'test')
    scan_type - viewing orientation plane ('axial', 'coronal', 'saggittal')
    use_negative - use series with negative label
    debug - debugging mode (only 10 series, fixed random states, chunks not shuffled)
    sampling - ratio of positive and negative chunks per series
    batch_size - batch size to use. If None, return the whole image for testing
    resize_shape - resize height and width to this shape
    num_axial_slices - number of slices per chunk (affects total number of chunks)
    chunk_stride - save memory by reducing overlap between chunks (TODO)
    crop_shape - center crop after augmentation (must be less than resize shape)
    use_3d - make compatible for 3D conv layers
    window_levels - specify different (window, levels) in the channel dimension
        (if not 3D, num_axial_slices must be 1)
    window_level_out - Sequence of length two, window width and level of the output of
         the intensity normalization process
    noise_factor - apply random noise to pixel values for regularization
    noise_dist - distribution to draw noise from
    rand_rotation - random in-plane rotations along z axis (in degrees)
    rand_roll - random rolling (offset pixels; edge pixels overflow to opposite side)
    rand_flip - random flipping each axis (assume indexed with channels-first)
    enable_augmentation - (bool) if False, all augmentation is disabled (overrides other parameters)
    return_mask - (bool), if True, masks are returned in the dict for positive cases
    pad_box - (int) add padding to the bounding boxes
    ohem_temperature - optional float. OHEM = online hard example mining. If None, no OHEM is performed. Otherwise,
        should be a positive float representing a temperature smooth the sampling distribution of the negatives. A low
        temperature (< 1) exaggerates the peaks of the distribution oversampling the hard cases. A high temperature
        (> 1) smooths the distribution, moving closer to uniform sampling.
    ohem_discount - float. The rate at which the OHEM scores change with each update. A discount of 1 implies that
        the score is never updated given new information. A discount of 0.0 means that all previous information is
        discarded on every update.

    '''
    def __init__(self,
                 data_path: Optional[str] = None,
                 csv_df: Optional[pd.DataFrame] = None,
                 data_split: str = 'train',
                 scan_type: str = 'axial',
                 use_negative: bool = True,
                 debug: bool = False,
                 sampling: MaybeInt = None,
                 batch_size: Optional[int] = 32,
                 resize_shape: Tuple[MaybeInt, MaybeInt] = (None, None),
                 num_axial_slices: int = 3,
                 chunk_stride: int = 1,
                 crop_shape: Tuple[MaybeInt, MaybeInt] = (None, None),
                 pad_box: int = 0,
                 use_3d: bool = False,
                 window_levels: Sequence[Tuple[int, int]] = ((1200, 500), ),
                 window_level_out: Tuple[float, float] = (1.0, 0.0),
                 noise_factor: MaybeFloat = None,
                 noise_dist: MaybeStr = None,
                 rand_rotation: MaybeInt = None,
                 rand_roll: Dict[str, MaybeInt] = {'h': None, 'w': None},
                 rand_flip: Sequence[bool] = [False, False, False],
                 enable_augmentation: bool = True,
                 return_mask: bool = False,
                 ohem_temperature: Optional[float] = None,
                 ohem_discount: float = 0.5
                 ) -> None:

        if not data_path:
            self.data_path = locations.get_preprocessed_dir(scan_type)
        else:
            self.data_path = Path(data_path)  # type: ignore
        if csv_df is not None:
            self.csv_df = csv_df  # type: ignore
        else:
            self.csv_path = locations.get_csv_dir() / 'splits'
            self.data_split = data_split
            self.scan_type = scan_type
            self.df_path = self.csv_path / f'{self.data_split}.csv'
            map(check_path_exists, [self.data_path, self.csv_path, self.df_path])
            self.csv_df = pd.read_csv(self.df_path)

        if not use_negative:
            self.csv_df = self.csv_df[self.csv_df.label == 1]

        self.mrn_accs = self.csv_df.mrn_acc.tolist()
        self.labels = self.csv_df.label.tolist()
        get_mrn: Callable[[str], str] = lambda s: s.split('_')[0]
        self.num_patients = np.unique(map(get_mrn, self.mrn_accs))

        self.debug = debug
        if self.debug:
            self.mrn_accs = self.mrn_accs[0:10]
            self.labels = self.labels[0:10]

        self.batch_size = batch_size
        if sampling is None or sampling == 0:
            self.sampling = 0
        elif sampling > 0 and self.batch_size is not None:
            self.sampling = sampling
            self.num_pos = int(self.batch_size / (self.sampling + 1))
            self.num_neg = self.batch_size - self.num_pos

        h_res, w_res = resize_shape
        h_crop, w_crop = crop_shape
        if h_res and h_crop and (h_res >= h_crop):
            raise ValueError('height: crop larger than resize')
        if w_res and w_crop and (w_res >= w_crop):
            raise ValueError('width: crop larger than resize')
        self.resize_shape = resize_shape
        self.crop_shape = crop_shape

        self.num_axial_slices = num_axial_slices
        self.chunk_stride = chunk_stride
        self.pad_box = pad_box
        self.end_offset = (self.num_axial_slices + 1) // 2
        self.start_offset = self.num_axial_slices - self.end_offset

        self.use_3d = use_3d
        if not self.use_3d and len(window_levels) > 1 and self.num_axial_slices > 1:
            raise ValueError('num_axial_slices must be 1 if using multiple windows in 2D')
        if len(window_levels) < 1:
            raise ValueError('Must provide at least one window level')
        self.window_levels = window_levels

        if len(window_level_out) != 2:
            raise ValueError('window level out should consist of a window width and level')
        self.wo = window_level_out[0]
        self.lo = window_level_out[1]

        self.noise_factor = noise_factor

        if noise_dist is None:
            self.noise_dist = np.random.normal
        elif noise_dist == 'normal':
            self.noise_dist = np.random.normal
        elif noise_dist == 'poisson':
            self.noise_dist = np.random.poisson
        else:
            raise ValueError('Not supported noise distribution')

        self.rand_rotation = rand_rotation
        self.rand_roll = rand_roll
        self.rand_flip = rand_flip
        self.enable_augmentation = enable_augmentation
        self.return_mask = return_mask

        self.ohem_temperature = ohem_temperature
        self.ohem_discount = ohem_discount
        if self.ohem_temperature is not None:
            if self.ohem_temperature <= 0.0:
                raise ValueError('OHEM temperature must be positive')
            if self.ohem_discount < 0.0 or ohem_discount >= 1.0:
                raise ValueError('OHEM discount should be between 0 (inclusive) and 1 (exclusive)')

            # A dictionary of mrn_accs, with each value being a dictionary of slice indices mapping to scores
            # Scores represent a weighted average of the highest score in the slice in recent epochs and are between
            # 0 and 1
            # A high score means a fracture was incorrectly detected in the slice recently, and it should therefore
            # be oversampled
            self.ohem_scores: Dict[str, Dict[int, float]] = defaultdict(lambda: defaultdict(lambda: 0.5))

    def safe_shuffle(self) -> None:
        if self.debug:
            random.setstate(0)
            np.random.set_state(0)
            torch.set_rng_state(0)
        temp = list(zip(self.mrn_accs, self.labels))
        random.shuffle(temp)
        self.mrn_accs, self.labels = zip(*temp)

    def load_vol(self, mrn_acc: str) -> Array:
        vol_path = self.data_path / 'ct' / f'{mrn_acc}.npy'
        check_path_exists(vol_path)
        vol = np.load(vol_path).astype(np.float)
        return vol

    def window_vol(self, vol: Tensor) -> Tensor:
        # For each window/level, clip at the values and normalize into the range (lo - wo / 2) to (lo + wo / 2)
        # Stack the different volumes along the channels axis
        vol = torch.cat(
            [(vol.clamp(l - w // 2, l + w // 2) - l) / (w / self.wo) + self.lo for w, l in self.window_levels],
            dim=1
        )
        return vol

    def resize(self, vol: Array, order: int) -> Array:
        f = functools.partial(
            transform.resize,
            order=order,
            clip=True,
            preserve_range=True,
            anti_aliasing=True
        )
        vol = np.moveaxis(vol, 0, -1)  # skimage expects depth-last
        vol = f(vol, self.resize_shape)
        return np.moveaxis(vol, -1, 0)

    def crop(self, vol: Array, mask: Optional[Array] = None) -> Tuple[Array, Optional[Array]]:
        h_crop, w_crop = self.crop_shape
        h_vol, w_vol = vol.shape[1:]
        if h_crop:
            assert h_crop <= h_vol, f'height: {h_crop} > {h_vol}'
            h_diff = h_vol - h_crop
            h_diff //= 2
            h_slice = slice(h_diff, h_diff + h_crop)
            vol = vol[:, h_slice, :]
            if mask is not None:
                mask = mask[:, h_slice, :]
        if w_crop:
            assert w_crop <= w_vol, f'width: {w_crop} > {w_vol}'
            w_diff = w_vol - w_crop
            w_diff //= 2
            w_slice = slice(w_diff, w_diff + w_crop)
            vol = vol[:, :, w_slice]
            if mask is not None:
                mask = mask[:, :, w_slice]

        if mask is not None:
            check_shape(vol, mask)

        return vol, mask

    def augment(self, vol: Array, mask: Optional[Array] = None) -> Tuple[Array, Optional[Array]]:
        f: Callable[[Array, float], Array] = functools.partial(
            transform.rotate,
            mode='constant',
            clip=True,
            preserve_range=True,
        )
        if self.rand_rotation:
            theta = np.random.uniform(-self.rand_rotation, self.rand_rotation)
            vol = f(np.moveaxis(vol, 0, -1), theta, order=1, cval=-1000)  # type: ignore
            vol = np.moveaxis(vol, -1, 0)
            if mask is not None and mask.sum():
                mask = f(np.moveaxis(mask, 0, -1), theta, order=0, cval=0)  # type: ignore
                mask = np.moveaxis(mask, -1, 0)
                check_shape(vol, mask)

        if self.rand_roll['h']:
            h_roll = round(self.rand_roll['h'] * np.random.random())
            vol = np.roll(vol, h_roll, axis=1)
            if mask is not None and mask.sum():
                mask = np.roll(mask, h_roll, axis=1)
                check_shape(vol, mask)

        if self.rand_roll['w']:
            w_roll = round(self.rand_roll['w'] * np.random.random())
            vol = np.roll(vol, w_roll, axis=2)
            if mask is not None and mask.sum():
                mask = np.roll(mask, w_roll, axis=2)
                check_shape(vol, mask)

        if sum(self.rand_flip):
            for i in range(vol.ndim):
                if np.random.randint(0, self.rand_flip[i] + 1):
                    vol = np.flip(vol, axis=i)
                    if mask is not None and mask.sum():
                        mask = np.flip(mask, axis=i)
                        check_shape(vol, mask)  # type: ignore

        if self.noise_factor:
            noise = self.noise_factor * self.noise_dist(size=vol.shape)
            vol += noise
            # np.add(vol, noise, out=vol, casting='unsafe')

        return vol, mask

    def __len__(self) -> int:
        return len(self.mrn_accs)

    def update_ohem_scores(self, mrn_acc: str, scores_per_slice: Mapping[int, float]) -> None:
        if self.ohem_temperature is not None:
            for s, score in scores_per_slice.items():
                self.ohem_scores[mrn_acc][s] = (
                    self.ohem_discount * self.ohem_scores[mrn_acc][s] + (1. - self.ohem_discount) * score
                )

    def ohem_sample_negatives(self, mrn_acc: str, idx: Sequence[int], k: int) -> List[int]:
        if self.ohem_temperature is not None:
            # Weight the negative slices according to the OHEM scores
            raw_ohem_scores = np.array([self.ohem_scores[mrn_acc][s] for s in list(idx)])

            # Run through softmax function to get the weights
            neg_weights = softmax(raw_ohem_scores / self.ohem_temperature)
        else:
            neg_weights = None
        return random.choices(list(idx), k=k, weights=neg_weights)

    def __getitem__(self, idx: Union[int, str]) -> Dict[str, Any]:
        if isinstance(idx, int):
            mrn_acc: str = self.mrn_accs[idx]
            int_idx = int(idx)
        elif isinstance(idx, str):
            mrn_acc = idx
            int_idx = self.mrn_accs.index(mrn_acc)
        label: Union[int, float] = self.labels[int_idx]
        neg = True if label == 0 else False

        vol: Array = self.load_vol(mrn_acc)
        d_vol, h_vol, w_vol = vol.shape[:3]

        w_res, h_res = self.resize_shape
        if (w_res and w_res != w_vol) or (h_res and h_res != h_vol):
            vol = self.resize(vol, order=1)

        if not neg:
            mask_path = self.data_path / 'mask' / f'{mrn_acc}.npy'
            check_path_exists(mask_path)
            mask = np.load(mask_path)
            d_mask, h_mask, w_mask = mask.shape

            if (w_res and w_res != w_mask) or (h_res and h_res != h_mask):
                mask = self.resize(mask, order=0)
            check_shape(vol, mask)

        # Pad in z if there are not enough slices
        if d_vol < self.num_axial_slices:
            pad_needed = self.num_axial_slices - d_vol
            pad_before = pad_needed // 2
            pad_after = pad_needed - pad_before
            vol = np.pad(vol, pad_width=[[pad_before, pad_after], [0, 0], [0, 0]], constant_values=-1000)
            d_vol = self.num_axial_slices
            if not neg:
                mask = np.pad(mask, pad_width=[[pad_before, pad_after], [0, 0], [0, 0]], constant_values=0)

        chunk_range = list(range(self.start_offset, d_vol - self.end_offset + 1))

        if self.batch_size is None:
            # If batch_size is None, return the entire image for testing
            if self.use_3d:
                # Use the chunk_stride parameter to pick a subset of the available chunks
                chunks = chunk_range[::self.chunk_stride]
                # If the end slices would otherwise be missed off, add another chunk at the very end
                if chunks[-1] != chunk_range[-1]:
                    chunks += [chunk_range[-1]]
            else:
                # Use every valid location
                chunks = chunk_range
        else:
            # Create a random batch
            if neg:
                chunks = self.ohem_sample_negatives(mrn_acc, idx=chunk_range, k=self.batch_size)
            else:
                if not self.sampling:
                    chunks = random.choices(chunk_range, k=self.batch_size)
                else:
                    # Split into positive and negative valid indices
                    pos_idx = set(np.where(mask.sum(axis=(1, 2)) > 0)[0]) & set(chunk_range)
                    neg_idx = set(chunk_range) - pos_idx
                    if len(pos_idx) == 0 or len(neg_idx) == 0:
                        chunks = random.choices(chunk_range, k=self.batch_size)
                    else:
                        chunks = self.ohem_sample_negatives(mrn_acc, idx=list(neg_idx), k=self.num_neg)
                        chunks += random.choices(list(pos_idx), k=self.num_pos)

        if self.debug or self.batch_size is None:
            chunks = sorted(chunks)
        else:
            random.shuffle(chunks)

        max_ann = 1  # used to get dimension of annotation
        imgs = []
        anns = []
        chunk_offsets = []
        if self.return_mask:
            masks = []

        try:
            for i in chunks:
                chunk_slices = slice(i - self.start_offset, i + self.end_offset)
                chunk = vol[chunk_slices]
                chunk_offsets.append(i - self.start_offset)
                if neg:
                    if self.enable_augmentation:
                        chunk, _ = self.augment(chunk)
                    if self.crop_shape[0] or self.crop_shape[1]:
                        chunk, _ = self.crop(chunk)
                else:
                    chunk_mask = mask[chunk_slices]
                    check_shape(chunk, chunk_mask)
                    if self.enable_augmentation:
                        chunk, chunk_mask = self.augment(chunk, chunk_mask)
                    if self.crop_shape[0] or self.crop_shape[1]:
                        chunk, chunk_mask = self.crop(chunk, chunk_mask)
                    check_shape(chunk, chunk_mask)

                if neg or not chunk_mask.sum():
                    box = torch.zeros((0, 8 if self.use_3d else 6))
                else:
                    fids = np.unique(chunk_mask)[1:]  # assume first index is background
                    box = torch.zeros((len(fids), 8 if self.use_3d else 6))

                    for j, fid in enumerate(fids):
                        seg = np.where(chunk_mask == fid)
                        if self.use_3d:
                            box[j, 0] = max(0, int(seg[0].min()) - self.pad_box)
                            box[j, 1] = max(0, int(seg[2].min()) - self.pad_box)
                            box[j, 2] = max(0, int(seg[1].min()) - self.pad_box)
                            box[j, 3] = min(int(seg[0].max()) + self.pad_box, chunk_mask.shape[0] - 1)
                            box[j, 4] = min(int(seg[2].max()) + self.pad_box, chunk_mask.shape[2] - 1)
                            box[j, 5] = min(int(seg[1].max()) + self.pad_box, chunk_mask.shape[1] - 1)
                            box[j, 6] = 0
                            box[j, 7] = fid
                        else:
                            box[j, 0] = max(0, int(seg[2].min()) - self.pad_box)
                            box[j, 1] = max(0, int(seg[1].min()) - self.pad_box)
                            box[j, 2] = min(int(seg[2].max()) + self.pad_box, chunk_mask.shape[2] - 1)
                            box[j, 3] = min(int(seg[1].max()) + self.pad_box, chunk_mask.shape[1] - 1)
                            box[j, 4] = 0
                            box[j, 5] = fid

                if self.use_3d:
                    chunk = chunk[np.newaxis]
                imgs.append(chunk)
                anns.append(box)
                if self.return_mask and not neg:
                    masks.append(chunk_mask)

                max_ann = max(max_ann, box.size(0))

            padded_anns = -torch.ones((len(anns), max_ann, 8 if self.use_3d else 6))
            for j, box in enumerate(anns):
                if box.size(0) > 0:
                    padded_anns[j, :box.size(0), :] = box

            ten_img = torch.from_numpy(np.stack(imgs))
            ten_img = self.window_vol(ten_img)

            ret: Dict[str, Any] = {
                'img': ten_img.contiguous(),
                'ann': padded_anns.contiguous(),
                'mrn_acc': mrn_acc,
                'negative': neg,
                'offsets': torch.tensor(chunk_offsets, dtype=torch.int64).contiguous(),
            }
            if self.return_mask:
                if neg:
                    ret['mask'] = None
                    # torch.zeros_like(ten_img)
                else:
                    ret['mask'] = torch.from_numpy(np.stack(masks))
            return ret

        except Exception as e:
            # Print some extra debugging info
            trace = traceback.format_exc()
            msg = f'\nCHUNK SHAPE:\t{chunk.shape}' \
                f'\nCHUNK:\t{i}' \
                f'\nLABEL:\t{label}'
            if not neg:
                msg += f'\nMASK SHAPE:\t{chunk_mask.shape}'
            print(trace + msg)
            raise(e)