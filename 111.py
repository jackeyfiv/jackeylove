import os

os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import sys

sys.path.append('comparison_methods')
import torch
from pathlib import Path
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from torch.utils.data.dataset import Dataset
import torch.nn as nn
import os

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
print('GPU num: ', torch.cuda.device_count())
from torch.utils.data import DataLoader
from torch import optim
from Stage_SSM import Stage_SSM
from dataloading import BasicDataset, CarvanaDataset
import time
import shutil


def shift9pos(input, h_shift_unit=1, w_shift_unit=1):
    input_pd = np.pad(input, ((h_shift_unit, h_shift_unit), (w_shift_unit, w_shift_unit)), mode='edge')
    input_pd = np.expand_dims(input_pd, axis=0)

    top = input_pd[:, :-2 * h_shift_unit, w_shift_unit:-w_shift_unit]
    bottom = input_pd[:, 2 * h_shift_unit:, w_shift_unit:-w_shift_unit]
    left = input_pd[:, h_shift_unit:-h_shift_unit, :-2 * w_shift_unit]
    right = input_pd[:, h_shift_unit:-h_shift_unit, 2 * w_shift_unit:]

    center = input_pd[:, h_shift_unit:-h_shift_unit, w_shift_unit:-w_shift_unit]

    bottom_right = input_pd[:, 2 * h_shift_unit:, 2 * w_shift_unit:]
    bottom_left = input_pd[:, 2 * h_shift_unit:, :-2 * w_shift_unit]
    top_right = input_pd[:, :-2 * h_shift_unit, 2 * w_shift_unit:]
    top_left = input_pd[:, :-2 * h_shift_unit, :-2 * w_shift_unit]

    shift_tensor = np.concatenate([top_left, top, top_right,
                                   left, center, right,
                                   bottom_left, bottom, bottom_right], axis=0)
    return shift_tensor


def init_spixel_grid(img_height, img_width, batch_size):
    curr_img_height = int(np.floor(img_height))
    curr_img_width = int(np.floor(img_width))

    all_h_coords = np.arange(0, curr_img_height, 1)
    all_w_coords = np.arange(0, curr_img_width, 1)
    curr_pxl_coord = np.array(np.meshgrid(all_h_coords, all_w_coords, indexing='ij'))
    coord_tensor = np.concatenate([curr_pxl_coord[1:2, :, :], curr_pxl_coord[:1, :, :]])
    all_XY_feat = (torch.from_numpy(np.tile(coord_tensor, (batch_size, 1, 1, 1)).astype(np.float32)).cpu())
    return all_XY_feat


def build_LABXY_feat(label_in, XY_feat):
    img_lab = label_in.clone().type(torch.float)
    b, _, curr_img_height, curr_img_width = XY_feat.shape
    scale_img = F.interpolate(img_lab, size=(curr_img_height, curr_img_width), mode='nearest')
    LABXY_feat = torch.cat([scale_img, XY_feat], dim=1)
    return LABXY_feat


def dice_loss_multiclass(pred, target, smooth=1e-5):
    target_onehot = F.one_hot(target, num_classes=3).permute(0, 3, 1, 2).float()
    intersection = (pred * target_onehot).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target_onehot.sum(dim=(2, 3))
    dice = (2. * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()


def dice_metric_multiclass(pred, target, smooth=1e-5):
    pred_onehot = F.one_hot(pred, num_classes=3).permute(0, 3, 1, 2).float()
    target_onehot = F.one_hot(target, num_classes=3).permute(0, 3, 1, 2).float()
    intersection = (pred_onehot * target_onehot).sum(dim=(2, 3))
    union = pred_onehot.sum(dim=(2, 3)) + target_onehot.sum(dim=(2, 3))
    dice_per_class = (2. * intersection + smooth) / (union + smooth)
    return dice_per_class.mean(dim=0)


def poolfeat(input, prob, sp_h=2, sp_w=2):
    def feat_prob_sum(feat_sum, prob_sum, shift_feat):
        feat_sum += shift_feat[:, :-1, :, :]
        prob_sum += shift_feat[:, -1:, :, :]
        return feat_sum, prob_sum

    b, _, h, w = input.shape
    h_shift_unit = 1
    w_shift_unit = 1
    p2d = (w_shift_unit, w_shift_unit, h_shift_unit, h_shift_unit)
    feat_ = torch.cat([input, torch.ones([b, 1, h, w]).cpu()], dim=1)
    prob_feat = F.avg_pool2d(feat_ * prob.narrow(1, 0, 1), kernel_size=(sp_h, sp_w), stride=(sp_h, sp_w))
    temp = F.pad(prob_feat, p2d, mode='constant', value=0)
    send_to_top_left = temp[:, :, 2 * h_shift_unit:, 2 * w_shift_unit:]
    feat_sum = send_to_top_left[:, :-1, :, :].clone()
    prob_sum = send_to_top_left[:, -1:, :, :].clone()

    prob_feat = F.avg_pool2d(feat_ * prob.narrow(1, 1, 1), kernel_size=(sp_h, sp_w), stride=(sp_h, sp_w))
    top = F.pad(prob_feat, p2d, mode='constant', value=0)[:, :, 2 * h_shift_unit:, w_shift_unit:-w_shift_unit]
    feat_sum, prob_sum = feat_prob_sum(feat_sum, prob_sum, top)

    prob_feat = F.avg_pool2d(feat_ * prob.narrow(1, 2, 1), kernel_size=(sp_h, sp_w), stride=(sp_h, sp_w))
    top_right = F.pad(prob_feat, p2d, mode='constant', value=0)[:, :, 2 * h_shift_unit:, :-2 * w_shift_unit]
    feat_sum, prob_sum = feat_prob_sum(feat_sum, prob_sum, top_right)

    prob_feat = F.avg_pool2d(feat_ * prob.narrow(1, 3, 1), kernel_size=(sp_h, sp_w), stride=(sp_h, sp_w))
    left = F.pad(prob_feat, p2d, mode='constant', value=0)[:, :, h_shift_unit:-h_shift_unit, 2 * w_shift_unit:]
    feat_sum, prob_sum = feat_prob_sum(feat_sum, prob_sum, left)

    prob_feat = F.avg_pool2d(feat_ * prob.narrow(1, 4, 1), kernel_size=(sp_h, sp_w), stride=(sp_h, sp_w))
    center = F.pad(prob_feat, p2d, mode='constant', value=0)[:, :, h_shift_unit:-h_shift_unit,
             w_shift_unit:-w_shift_unit]
    feat_sum, prob_sum = feat_prob_sum(feat_sum, prob_sum, center)

    prob_feat = F.avg_pool2d(feat_ * prob.narrow(1, 5, 1), kernel_size=(sp_h, sp_w), stride=(sp_h, sp_w))
    right = F.pad(prob_feat, p2d, mode='constant', value=0)[:, :, h_shift_unit:-h_shift_unit, :-2 * w_shift_unit]
    feat_sum, prob_sum = feat_prob_sum(feat_sum, prob_sum, right)

    prob_feat = F.avg_pool2d(feat_ * prob.narrow(1, 6, 1), kernel_size=(sp_h, sp_w), stride=(sp_h, sp_w))
    bottom_left = F.pad(prob_feat, p2d, mode='constant', value=0)[:, :, :-2 * h_shift_unit, 2 * w_shift_unit:]
    feat_sum, prob_sum = feat_prob_sum(feat_sum, prob_sum, bottom_left)

    prob_feat = F.avg_pool2d(feat_ * prob.narrow(1, 7, 1), kernel_size=(sp_h, sp_w), stride=(sp_h, sp_w))
    bottom = F.pad(prob_feat, p2d, mode='constant', value=0)[:, :, :-2 * h_shift_unit, w_shift_unit:-w_shift_unit]
    feat_sum, prob_sum = feat_prob_sum(feat_sum, prob_sum, bottom)

    prob_feat = F.avg_pool2d(feat_ * prob.narrow(1, 8, 1), kernel_size=(sp_h, sp_w), stride=(sp_h, sp_w))
    bottom_right = F.pad(prob_feat, p2d, mode='constant', value=0)[:, :, :-2 * h_shift_unit, :-2 * w_shift_unit]
    feat_sum, prob_sum = feat_prob_sum(feat_sum, prob_sum, bottom_right)

    pooled_feat = feat_sum / (prob_sum + 1e-8)
    return pooled_feat


def upfeat(input, prob, up_h=2, up_w=2):
    b, c, h, w = input.shape
    h_shift = 1
    w_shift = 1
    p2d = (w_shift, w_shift, h_shift, h_shift)
    feat_pd = F.pad(input, p2d, mode='constant', value=0)

    gt_frm_top_left = F.interpolate(feat_pd[:, :, :-2 * h_shift, :-2 * w_shift], size=(h * up_h, w * up_w),
                                    mode='nearest')
    feat_sum = gt_frm_top_left * prob.narrow(1, 0, 1)

    top = F.interpolate(feat_pd[:, :, :-2 * h_shift, w_shift:-w_shift], size=(h * up_h, w * up_w), mode='nearest')
    feat_sum += top * prob.narrow(1, 1, 1)

    top_right = F.interpolate(feat_pd[:, :, :-2 * h_shift, 2 * w_shift:], size=(h * up_h, w * up_w), mode='nearest')
    feat_sum += top_right * prob.narrow(1, 2, 1)

    left = F.interpolate(feat_pd[:, :, h_shift:-w_shift, :-2 * w_shift], size=(h * up_h, w * up_w), mode='nearest')
    feat_sum += left * prob.narrow(1, 3, 1)

    center = F.interpolate(input, (h * up_h, w * up_w), mode='nearest')
    feat_sum += center * prob.narrow(1, 4, 1)

    right = F.interpolate(feat_pd[:, :, h_shift:-w_shift, 2 * w_shift:], size=(h * up_h, w * up_w), mode='nearest')
    feat_sum += right * prob.narrow(1, 5, 1)

    bottom_left = F.interpolate(feat_pd[:, :, 2 * h_shift:, :-2 * w_shift], size=(h * up_h, w * up_w), mode='nearest')
    feat_sum += bottom_left * prob.narrow(1, 6, 1)

    bottom = F.interpolate(feat_pd[:, :, 2 * h_shift:, w_shift:-w_shift], size=(h * up_h, w * up_w), mode='nearest')
    feat_sum += bottom * prob.narrow(1, 7, 1)

    bottom_right = F.interpolate(feat_pd[:, :, 2 * h_shift:, 2 * w_shift:], size=(h * up_h, w * up_w), mode='nearest')
    feat_sum += bottom_right * prob.narrow(1, 8, 1)

    return feat_sum


def compute_semantic_pos_loss(prob_in, labxy_feat, pos_weight=0.003, kernel_size=16):
    S = kernel_size
    m = pos_weight
    prob = prob_in.clone()
    b, c, h, w = labxy_feat.shape
    pooled_labxy = poolfeat(labxy_feat, prob, kernel_size, kernel_size)
    reconstr_feat = upfeat(pooled_labxy, prob, kernel_size, kernel_size)
    loss_map = reconstr_feat[:, -2:, :, :] - labxy_feat[:, -2:, :, :]
    logit = torch.log(reconstr_feat[:, :-2, :, :] + 1e-8)
    loss_sem = - torch.sum(logit * labxy_feat[:, :-2, :, :]) / b
    loss_pos = torch.norm(loss_map, p=2, dim=1).sum() / b * m / S
    loss_sum = 0.005 * (loss_sem + loss_pos)
    loss_sem_sum = 0.005 * loss_sem
    loss_pos_sum = 0.005 * loss_pos
    return loss_sum, loss_sem_sum, loss_pos_sum


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def custom_collate_fn(batch):
    valid_batch = []
    for item in batch:
        if isinstance(item, dict) and 'image' in item and 'mask' in item:
            mask = item['mask']
            if mask.dim() == 2:
                mask = mask.unsqueeze(0)
            valid_batch.append({'image': item['image'], 'mask': mask})
    if not valid_batch:
        raise RuntimeError("No valid items in batch!")
    images = torch.stack([item['image'] for item in valid_batch])
    masks = torch.stack([item['mask'] for item in valid_batch])
    return {'image': images, 'mask': masks}


def dice_loss_binary(pred, target, smooth=1e-5):
    pred = torch.sigmoid(pred)
    intersection = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    dice = (2. * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()


def dice_metric_binary(pred, target, smooth=1e-5):
    intersection = (pred * target).sum(dim=(1, 2))
    total = pred.sum(dim=(1, 2)) + target.sum(dim=(1, 2))
    dice = (2. * intersection + smooth) / (total + smooth)
    return dice.mean()


# =================== 新增的验证函数 ===================
def evaluate_binary(model, val_loader):
    model.eval()
    total_dice = 0.0
    num_batches = len(val_loader)

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            images = batch['image'].to(device)
            masks = batch['mask'].to(device).float()

            masks_pred, _ = model(images)
            pred_classes = (torch.sigmoid(masks_pred[:, 1]) > 0.5).float()

            dice = dice_metric_binary(pred_classes, masks.squeeze(1))
            total_dice += dice.item()

    avg_dice = total_dice / num_batches
    model.train()
    return avg_dice


# =================== 修改后的训练函数 ===================
def train_model_binary(model, criterion, optimizer, scheduler, train_loader, val_loader,
                       num_epochs, total_epoch, model_name, state_save_path, state_load_path=None):
    if state_load_path is not None:
        model.load_state_dict(torch.load(state_load_path))

    num_step = len(train_loader)
    best_val_dice = 0.0
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    best_model_path = os.path.join(state_save_path, f"{model_name}_best_{timestamp}.pth")

    def init_spixel_grid(img_height, img_width, batch_size):
        curr_img_height = int(np.floor(img_height))
        curr_img_width = int(np.floor(img_width))
        all_h_coords = np.arange(0, curr_img_height, 1)
        all_w_coords = np.arange(0, curr_img_width, 1)
        curr_pxl_coord = np.array(np.meshgrid(all_h_coords, all_w_coords, indexing='ij'))
        coord_tensor = np.concatenate([curr_pxl_coord[1:2, :, :], curr_pxl_coord[:1, :, :]])
        all_XY_feat = torch.from_numpy(np.tile(coord_tensor, (batch_size, 1, 1, 1)).astype(np.float32))
        return all_XY_feat.to(device)

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        train_dice_score = 0.0
        batch_size = train_loader.batch_size

        xy_feat1 = init_spixel_grid(64, 64, batch_size)
        xy_feat2 = init_spixel_grid(32, 32, batch_size)
        xy_feat3 = init_spixel_grid(16, 16, batch_size)
        xy_feat4 = init_spixel_grid(8, 8, batch_size)

        with tqdm(total=num_step, desc=f'Epoch {epoch + 1}/{num_epochs}', postfix=dict, mininterval=0.3) as pbar:
            for iteration, batch in enumerate(train_loader):
                images = batch['image'].to(device)
                masks = batch['mask'].to(device).float()
                batch_step = images.shape[0]

                if iteration == num_step - 1:
                    xy_feat1 = init_spixel_grid(64, 64, batch_step)
                    xy_feat2 = init_spixel_grid(32, 32, batch_step)
                    xy_feat3 = init_spixel_grid(16, 16, batch_step)
                    xy_feat4 = init_spixel_grid(8, 8, batch_step)

                masks1 = F.interpolate(masks, size=(64, 64), mode='nearest')
                masks2 = F.interpolate(masks, size=(32, 32), mode='nearest')
                masks3 = F.interpolate(masks, size=(16, 16), mode='nearest')
                masks4 = F.interpolate(masks, size=(8, 8), mode='nearest')

                LABXY_feat_tensor1 = build_LABXY_feat(masks1, xy_feat1)
                LABXY_feat_tensor2 = build_LABXY_feat(masks2, xy_feat2)
                LABXY_feat_tensor3 = build_LABXY_feat(masks3, xy_feat3)
                LABXY_feat_tensor4 = build_LABXY_feat(masks4, xy_feat4)

                masks_pred, Q_prob_collect = model(images)

                LABXY_feat_tensor1 = F.interpolate(LABXY_feat_tensor1, size=Q_prob_collect[0].shape[2:],
                                                   mode='bilinear', align_corners=False)
                LABXY_feat_tensor2 = F.interpolate(LABXY_feat_tensor2, size=Q_prob_collect[1].shape[2:],
                                                   mode='bilinear', align_corners=False)
                LABXY_feat_tensor3 = F.interpolate(LABXY_feat_tensor3, size=Q_prob_collect[2].shape[2:],
                                                   mode='bilinear', align_corners=False)
                LABXY_feat_tensor4 = F.interpolate(LABXY_feat_tensor4, size=Q_prob_collect[3].shape[2:],
                                                   mode='bilinear', align_corners=False)

                slic_loss1, _, _ = compute_semantic_pos_loss(Q_prob_collect[0], LABXY_feat_tensor1,
                                                             pos_weight=0.003, kernel_size=2)
                slic_loss2, _, _ = compute_semantic_pos_loss(Q_prob_collect[1], LABXY_feat_tensor2,
                                                             pos_weight=0.003, kernel_size=2)
                slic_loss3, _, _ = compute_semantic_pos_loss(Q_prob_collect[2], LABXY_feat_tensor3,
                                                             pos_weight=0.003, kernel_size=2)
                slic_loss4, _, _ = compute_semantic_pos_loss(Q_prob_collect[3], LABXY_feat_tensor4,
                                                             pos_weight=0.003, kernel_size=1)

                loss_value = criterion(masks_pred, masks.long().squeeze(1)) + \
                             dice_loss_binary(masks_pred[:, 1:2, :, :], masks)
                loss_sum = loss_value + (slic_loss1 + slic_loss2 + slic_loss3 + slic_loss4) * 0.2

                optimizer.zero_grad()
                loss_sum.backward()
                optimizer.step()
                epoch_loss += loss_sum.item()

                pred_classes = (torch.sigmoid(masks_pred[:, 1]) > 0.5).float()
                batch_dice = dice_metric_binary(pred_classes, masks.squeeze(1))
                train_dice_score += batch_dice.item()

                pbar.set_postfix(**{
                    'train_loss': epoch_loss / (iteration + 1),
                    'lr': get_lr(optimizer),
                    'train_dice': train_dice_score / (iteration + 1)
                })
                pbar.update(1)

        scheduler.step()

        # ================ 每个epoch结束后进行验证 ================
        val_dice = evaluate_binary(model, val_loader)
        print(f"Epoch {epoch + 1}/{num_epochs} - Validation Dice: {val_dice:.4f}")

        # 保存当前epoch的模型
        epoch_model_path = os.path.join(state_save_path, f"{model_name}_epoch{epoch + 1}_{total_epoch}.pth")
        torch.save(model, epoch_model_path)
        print(f"Saved model for epoch {epoch + 1} to {epoch_model_path}")

        # 保存最佳模型
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            shutil.copyfile(epoch_model_path, best_model_path)
            print(f"New best model! Validation Dice: {best_val_dice:.4f}, saved to {best_model_path}")

    print(f"Training completed. Best Validation Dice: {best_val_dice:.4f}")
    return best_val_dice


if __name__ == '__main__':
    model = Stage_SSM(num_class=2)
    model = model.to(device)

    # 训练集路径
    dir_img_train = Path(r'E:\1\camus\train\img')
    dir_mask_train = Path(r'E:\1\camus\train\mask')

    # 验证集路径
    dir_img_val = Path(r'E:\1\camus\val\img')
    dir_mask_val = Path(r'E:\1\camus\val\mask')

    # 创建数据集
    try:
        train_set = CarvanaDataset(dir_img_train, dir_mask_train, 1)
        val_set = CarvanaDataset(dir_img_val, dir_mask_val, 1)
    except (AssertionError, RuntimeError):
        train_set = BasicDataset(dir_img_train, dir_mask_train, 1)
        val_set = BasicDataset(dir_img_val, dir_mask_val, 1)

    # 检查数据集
    for i, name in zip([0, len(train_set) - 1], ["first", "last"]):
        item = train_set[i]
        print(f"Train {name} sample:")
        print(f"  Image shape: {item['image'].shape}")
        print(f"  Mask shape: {item['mask'].shape}")
        print(f"  Unique mask values: {torch.unique(item['mask'])}")

    for i, name in zip([0, len(val_set) - 1], ["first", "last"]):
        item = val_set[i]
        print(f"Val {name} sample:")
        print(f"  Image shape: {item['image'].shape}")
        print(f"  Mask shape: {item['mask'].shape}")
        print(f"  Unique mask values: {torch.unique(item['mask'])}")

    # 数据加载器参数
    batch_size = 1
    loader_args = dict(
        batch_size=batch_size,
        num_workers=0,
        pin_memory=True,
        collate_fn=custom_collate_fn
    )

    # 创建数据加载器
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, **loader_args)

    # 训练参数
    learning_rate = 0.000075
    num_epochs = 20
    total_epoch = 20
    model_name = 'Stage_SSM_all'
    state_save_path = r'C:\Users\74424\PycharmProjects\SSM1\checkpoints/'

    # 确保保存路径存在
    os.makedirs(state_save_path, exist_ok=True)

    # 优化器和学习率调度器
    optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=1e-8, momentum=0.9)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.91)
    criterion = nn.CrossEntropyLoss()

    # 开始训练
    best_val_dice = train_model_binary(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=lr_scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs,
        total_epoch=total_epoch,
        model_name=model_name,
        state_save_path=state_save_path,
        state_load_path=None
    )

    print(f"Training completed. Best validation Dice: {best_val_dice:.4f}")