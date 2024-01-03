from torch import nn
import torch
from network import Unet
from utils import ProjOperator, FBPOperator, compare, compute_psnr
from torch.utils.data import Dataset, DataLoader
import logging
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import glob
import os
import pydicom
from torch.utils.data import Subset
from skimage.restoration import denoise_tv_chambolle as TV

device = torch.device("cuda:0")
torch.cuda.set_device(0)


angles=64
size = 512
radon_pro = ProjOperator(N=size, M=size, pixel_size_x=0.15, pixel_size_y=0.15,
                         det_pixels=624, det_pixel_size=0.2, angles=angles, src_origin=950,
                         det_origin=200)
radon_pro_label = ProjOperator(N=size, M=size, pixel_size_x=0.15, pixel_size_y=0.15,
                               det_pixels=624, det_pixel_size=0.2, angles=192, src_origin=950,
                               det_origin=200)
radon_pro_label_360 = ProjOperator(N=size, M=size, pixel_size_x=0.15, pixel_size_y=0.15,
                                   det_pixels=624, det_pixel_size=0.2, angles=360, src_origin=950,
                                   det_origin=200)
fbp_op = FBPOperator(N=size, M=size, pixel_size_x=0.15, pixel_size_y=0.15,
                     det_pixels=624, det_pixel_size=0.2, angles=192, src_origin=950,
                     det_origin=200, filter_type='Ram-Lak', frequency_scaling=0.7)
fbp_op_360 = FBPOperator(N=size, M=size, pixel_size_x=0.15, pixel_size_y=0.15,
                         det_pixels=624, det_pixel_size=0.2, angles=360, src_origin=950,
                         det_origin=200, filter_type='Ram-Lak', frequency_scaling=1)
fbp_op_64 = FBPOperator(N=size, M=size, pixel_size_x=0.15, pixel_size_y=0.15,
                        det_pixels=624, det_pixel_size=0.2, angles=angles, src_origin=950,
                        det_origin=200, filter_type='Ram-Lak', frequency_scaling=0.7)


class DicomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the DICOM images.
            transform (callable, optional): Optional transform to be applied on an image.
            size (int, optional): The size to which the images will be resized.

        """
        self.root_dir = root_dir
        self.transform = transform
        self.file_list = glob.glob(os.path.join(self.root_dir, '**/*.IMA'), recursive=True)


    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.file_list[idx]
        image = pydicom.read_file(img_path).pixel_array
        image = np.float32(image)
        image = image / np.max(image)
        image[image < 0] = 0

        # image = np.expand_dims(image, axis=0)  # add channel dimension
        # image = torch.from_numpy(image)

        if self.transform:
            image = self.transform(image)


        return image


Unet = Unet(img_channel=1, width=32)
Unet.to(device)
Unet = nn.DataParallel(Unet)
pretrained_state_dict = torch.load('/home/w/d2gan_end_to_end/train_odl/resutlt_60_512/my_model_0102.pth')
Unet.load_state_dict(pretrained_state_dict)

val_dataset = DicomDataset('/home/student1/wujia/d2gan_end_to_end/raw_data/test/L***/full_3mm')
print(len(val_dataset))
single_image_dataset = Subset(val_dataset, [0])


def val_fn(validlow):
    validlow = np.reshape(validlow, (1, 1, size, size))
    validlow = torch.from_numpy(validlow).cuda()
    with torch.no_grad():
        proj_64 = radon_pro(validlow)
        proj_192 = F.interpolate(proj_64, size=(192, proj_64.shape[3]))
        unet_output = Unet(proj_192)
        ct_recon = fbp_op(unet_output)
        unet_output2 = Unet(ct_recon)
        # unet_output2 = unet_output2 - fbp_op_64(radon_pro(unet_output2) - proj_64)
    output = unet_output2.cpu().numpy()[0, 0, :, :]
    return output

def val_pfn(validlow):
    validlow = np.reshape(validlow, (1, 1, size, size))
    validlow = torch.from_numpy(validlow).cuda()
    with torch.no_grad():
        proj_64 = radon_pro(validlow)
        proj_192 = F.interpolate(proj_64, size=(192, proj_64.shape[3]))
        unet_output = Unet(proj_192)
        ct_recon = fbp_op(unet_output)
        unet_output2 = Unet(ct_recon)
    prj_out = unet_output.cpu().numpy()[0, 0, :, :]

    return prj_out

Pro = np.ones([192, 624])
aat = radon_pro_label(fbp_op(Pro))

Img = np.ones([size, size])
ata = fbp_op_64(radon_pro(Img))

ata2 = fbp_op(radon_pro_label(Img))

lambda1 = 0.05  # % The reconstruction parameters
lambda2 = 0.14
lambda3 = 0.6
deta1 = 0.003
deta2 = 0.003
Iter = 2000
loss = []
nw = size
nh = size
results = np.zeros([nw, nh, Iter])
reconstructions = np.zeros([nw, nh])
g_FBP1 = np.zeros([1, 1, nw, nh])
g_FBP2 = np.zeros([1, 1, nw, nh])
g_FBP22 = np.zeros([1, 1, nw, nh])
u = np.zeros([nw, nh])
v = np.zeros([nw, nh])
f1 = np.zeros([nw, nh])
f2 = np.zeros([nw, nh])

for idx, data in enumerate(single_image_dataset):
    reference = data
    print(idx)
    for iters in range(2000):
        if iters == 0:
            Proj1 = radon_pro(reference)
            Proj11 = radon_pro_label(reference)
            rawRec = fbp_op_64(Proj1)
            Proj2 = val_pfn(reference)
            g_FBP2 = val_fn(reference)

            Rnet = g_FBP2
            compare(reference, Rnet)
        else:
            resimage1 = fbp_op_64(radon_pro(g_FBP2) - Proj1)
            g_FBP2 = g_FBP2 - 0.1 * (
                    resimage1  + lambda3 * (
                    g_FBP2 - Rnet - v - f2)) / (ata + lambda3)

            g_FBP3 = g_FBP2 - Rnet - f2

            v1 = TV(g_FBP3, weight=deta2, n_iter_max=100)
            v = v1
            f2 = f2 + 1.0 * (v - g_FBP2 + Rnet)
            if iters % 10 == 0:
                compare(reference, g_FBP2)
