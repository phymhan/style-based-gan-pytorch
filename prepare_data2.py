import argparse
from io import BytesIO
import multiprocessing
from functools import partial

from PIL import Image
import lmdb
from tqdm import tqdm
from torchvision import datasets
from torchvision.transforms import functional as trans_fn
import os
from torch.utils.data import Dataset


def resize_and_convert(img, size, quality=100):
    img = trans_fn.resize(img, size)
    img = trans_fn.center_crop(img, size)
    buffer = BytesIO()
    img.save(buffer, format='jpeg', quality=quality)
    val = buffer.getvalue()

    return val


def resize_multiple(img, sizes=(8, 16, 32, 64, 128, 256, 512), quality=100):
    imgs = []

    for size in sizes:
        imgs.append(resize_and_convert(img, size, quality))

    return imgs


def resize_worker(img_file, sizes):
    i, file = img_file
    img = Image.open(file)
    img = img.convert('RGB')
    out = resize_multiple(img, sizes=sizes)
    return i, out


def prepare(txn, imglist, n_worker, sizes=(8, 16, 32, 64, 128, 256, 512)):
    resize_fn = partial(resize_worker, sizes=sizes)

    files = sorted(imglist, key=lambda x: x[0])
    files = [(i, file) for i, (file, label) in enumerate(files)]
    total = 0

    with multiprocessing.Pool(n_worker) as pool:
        for i, imgs in tqdm(pool.imap_unordered(resize_fn, files)):
            for size, img in zip(sizes, imgs):
                key = f'{size}-{str(i).zfill(5)}'.encode('utf-8')
                txn.put(key, img)

            total += 1

        txn.put('length'.encode('utf-8'), str(total).encode('utf-8'))


class ImageList(Dataset):
    def __init__(self, dataroot, sourcefile):
        self.root = dataroot
        with open(sourcefile, 'r') as f:
            self.imgs = [line.rstrip('\n').split() for line in f.readlines()]

    def __getitem__(self, index):
        print(f'----->>>>> {self.imgs[index]}')
        img = os.path.join(self.root, self.imgs[index][0])
        lbl = int(self.imgs[index][1])
        return img, lbl

    def __len__(self):
        return len(self.imgs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=str)
    parser.add_argument('--n_worker', type=int, default=8)
    parser.add_argument('--sourcefile', type=str, default='train.txt')
    parser.add_argument('path', type=str)

    args = parser.parse_args()

    imglist = []
    with open(args.sourcefile, 'r') as f:
        for line in f.readlines():
            imglist.append((os.path.join(args.path, line.rstrip().split()[0]), int(line.rstrip().split()[1])))

    with lmdb.open(args.out, map_size=1024 ** 4, readahead=False) as env:
        with env.begin(write=True) as txn:
            prepare(txn, imglist, args.n_worker)
