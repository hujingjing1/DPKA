from torch import nn
import torch
from network import Network
from utils import ProjOperator, FBPOperator, compare, DicomDataset, compute_psnr
from torch.utils.data import Dataset, DataLoader
import logging
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import pandas as pd
import pydicom
import glob
import os


device = torch.device("cuda:0")
torch.cuda.set_device(0)

Unet = Network(img_channel=1, width=32)
Unet.to(device)
Unet = nn.DataParallel(Unet)
# pretrained_state_dict = torch.load('/home/w/d2gan_end_to_end/train_odl/resutlt_60_512/my_model_pic_0104.pth')
# Unet.load_state_dict(pretrained_state_dict)
size=512
angles=64
radon_pro = ProjOperator(N=size, M=size, pixel_size_x=0.15, pixel_size_y=0.15,
                 det_pixels=624, det_pixel_size=0.2, angles=angles, src_origin=950,
                 det_origin=200)
radon_pro_label = ProjOperator(N=size, M=size, pixel_size_x=0.15, pixel_size_y=0.15,
                 det_pixels=624, det_pixel_size=0.2, angles=192, src_origin=950,
                 det_origin=200)
fbp_op = FBPOperator(N=size, M=size, pixel_size_x=0.15, pixel_size_y=0.15,
                 det_pixels=624, det_pixel_size=0.2, angles=192, src_origin=950,
                 det_origin=200, filter_type='Ram-Lak', frequency_scaling=0.7)

fbp_op_angel = FBPOperator(N=size, M=size, pixel_size_x=0.15, pixel_size_y=0.15,
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
        self.size = size
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

        image = np.expand_dims(image, axis=0)  # add channel dimension
        image = torch.from_numpy(image)

        if self.transform:
            image = self.transform(image)


        return image
    
dataset = DicomDataset('/home/w/d2gan_end_to_end/raw_data/full_3mm/L***/full_3mm')
print(len(dataset))

val_dataset = DicomDataset('/home/student1/wujia/d2gan_end_to_end/raw_data/test/L***/full_3mm')
print(len(val_dataset))

loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=True, num_workers=4)

learning_rate = 5e-4  # or any other value you want to set
optimizer = torch.optim.AdamW(Unet.parameters(), weight_decay=1e-5, lr=learning_rate, betas=(0.9, 0.999))
t_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-5)


# Loss functions
l1_loss = torch.nn.L1Loss()
l2_loss = torch.nn.MSELoss()
mse_loss = torch.nn.MSELoss()

train_losses, val_losses = [], []
train_mses, val_mses = [], []
train_psnrs, val_psnrs = [], []
best_train_loss = float('inf')
best_train_image = None
best_val_loss = float('inf')
best_val_image = None

# Start training loop
for epoch in range(100):
    print(f"Epoch {epoch + 1}")
    Unet.train()
    loop = tqdm(enumerate(loader), total=len(loader), leave=False)
    running_total_loss = 0
    running_mse_value = 0
    running_psnr_value = 0
    for j, data in loop:
        # Move data to the device
        data = data.to(device)
        # t = diffusion.sample_timesteps(data.shape[0]).to(device)

        # Forward pass through radon_net_project to get 64 projections
        proj_64 = radon_pro(data)
        proj_192_lable = radon_pro_label(data)
        # print( proj_192_lable.shape)

        # Interpolate 64 projections to 196
        proj_192 = F.interpolate(proj_64, size=(192, proj_64.shape[3]))
        # print(proj_192.shape)

        # Pass the interpolated projections through the UNet
        unet_output = Unet(proj_192)
        # print(unet_output.shape)

        # Compute the first L1 loss
        loss1 = l1_loss(proj_192_lable, unet_output)

        # Pass the UNet logs through the fbp_net to get the reconstructed CT
        ct_recon = fbp_op(unet_output)

        # Compute the second L2 loss
        loss2 = l1_loss(data, ct_recon)

        # Pass the reconstructed CT back through the UNet
        unet_output2 = Unet(ct_recon)
        # unet_output2 = unet_output2 - fbp_op_angel(radon_pro(unet_output2) - proj_64)


        # Compute the third L1 loss
        loss3 = l1_loss(unet_output2, data)

        # Combine the losses
        total_loss = 0.1 * loss1 + 0.1 * loss2 + loss3


        # total_loss = mse_loss(data, unet_output2)


        # Backward pass and optimization
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # Print out PSNR and MSE
        with torch.no_grad():
            mse_recon, _, psnr_recon = compare(data.detach().cpu().numpy(), unet_output2.detach().cpu().numpy(), verbose=False)


            running_total_loss += total_loss.item()
            running_mse_value +=  mse_recon.item()
            running_psnr_value += psnr_recon.item()
        # Set progress bar description and postfix
        loop.set_description(f"Epoch [{epoch + 1}/100]")
        loop.set_postfix(avg_train_loss=running_total_loss / (j + 1), avg_train_mse=running_mse_value / (j + 1),
                         avg_train_psnr=running_psnr_value / (j + 1))
    # t_scheduler.step()

    # Begin validation
    Unet.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # Disable gradient computation during validation
        running_val_loss = 0
        running_val_mse = 0
        running_val_psnr = 0
        val_loop = tqdm(enumerate(val_loader), total=len(val_loader), leave=False)
        for i, val_data in val_loop:
            val_data = val_data.to(device)
            # t = diffusion.sample_timesteps(val_data.shape[0]).to(device)

            # Your forward pass and loss computation code here for validation...
            val_proj_64 = radon_pro(val_data)
            val_proj_192_lable = radon_pro_label(val_data)
            val_proj_192 = F.interpolate(val_proj_64, size=(192, val_proj_64.shape[3]))
            val_unet_output = Unet(val_proj_192)
            # print(val_unet_output.shape)
            val_loss1 = l1_loss(val_proj_192_lable, val_unet_output)
            val_ct_recon = fbp_op(val_unet_output)
            val_loss2 = l1_loss(val_data, val_ct_recon)
            val_unet_output2 = Unet(val_ct_recon)
            # val_unet_output2 = val_unet_output2 - fbp_op_angel(radon_pro(val_unet_output2) - val_proj_64)

            val_loss3 = l1_loss(val_unet_output2, val_data)
            val_total_loss = 0.1 * val_loss1 + 0.1 * val_loss2 + val_loss3

            # val_total_loss = mse_loss(val_data, val_unet_output2)
            with torch.no_grad():
                mse_recon, _, psnr_recon = compare(val_data.detach().cpu().numpy(), val_unet_output2.detach().cpu().numpy(), verbose=False)


                running_val_loss += val_total_loss.item()
                running_val_mse +=  mse_recon.item()
                running_val_psnr += psnr_recon.item()

            # 更新进度条
            val_loop.set_description(f'Validation Epoch [{epoch + 1}/100]')
            val_loop.set_postfix(avg_val_loss=running_val_loss / (i + 1), avg_val_mse=running_val_mse / (i + 1),
                                 avg_val_psnr=running_val_psnr / (i + 1))

        avg_train_loss = running_total_loss / len(loader)
        avg_train_mse = running_mse_value / len(loader)
        avg_train_psnr = running_psnr_value / len(loader)
        avg_val_loss = running_val_loss / len(val_loader)
        avg_val_mse = running_val_mse / len(val_loader)
        avg_val_psnr = running_val_psnr / len(val_loader)
        print(f'Train loss: {avg_train_loss}, Train MSE: {avg_train_mse}, Train PSNR: {avg_train_psnr}')
        print(f'Validation loss: {avg_val_loss}, MSE: {avg_val_mse}, PSNR: {avg_val_psnr}')

        # Add current metrics to history
        train_losses.append(avg_train_loss)
        train_mses.append(avg_train_mse)
        train_psnrs.append(avg_train_psnr)
        val_losses.append(avg_val_loss)
        val_mses.append(avg_val_mse)
        val_psnrs.append(avg_val_psnr)

    logging.info(f"Epoch: {epoch + 1}")
    logging.info(f"Validation Loss: {avg_val_loss}, MSE: {avg_val_mse}, PSNR: {avg_val_psnr}")

    # Save the model if it has the best validation loss so far

    if avg_train_loss < best_train_loss:
        best_train_loss = avg_train_loss
        best_train_image = unet_output2[0].detach().cpu().numpy()
        data_real_train = data[0].cpu().numpy()
        # Calculate PSNR and MSE
        best_psnr_train = compute_psnr(best_train_image, data_real_train)
        best_mse_train = F.mse_loss(data, unet_output2)

        # Get the last four characters of the image name
        image_name_train = str(data)

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(Unet.state_dict(), '/home/w/d2gan_end_to_end/train_odl/resutlt_60_512/my_model_0110.pth')
        best_val_image = val_unet_output2.cpu().numpy()
        data_phantom = val_data.cpu().numpy()
        difference_image = np.abs(best_val_image - data_phantom)
        # Calculate PSNR and MSE
        best_psnr = compute_psnr(best_val_image, data_phantom)
        best_mse = F.mse_loss(val_data, val_unet_output2)

        # Get the last four characters of the image name
        image_name = str(val_data)

    # Create a DataFrame from the lists of metrics
    metrics_df = pd.DataFrame({
        'train_loss': train_losses,
        'train_mse': train_mses,
        'train_psnr': train_psnrs,
        'val_loss': val_losses,
        'val_mse': val_mses,
        'val_psnr': val_psnrs
    })

    # Save the DataFrame to a csv file with index
    metrics_df.to_csv('/home/w/d2gan_end_to_end/train_odl/resutlt_60_512/my_model_pic_0104.csv')