import torch
import utils.logging
import os
import torchvision
from torchvision.transforms.functional import crop


# This script is adapted from the following repository: https://github.com/ermongroup/ddim

def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a    # [n, 1, 1, 1]


def data_transform(X):
    return 2 * X - 1.0


def inverse_data_transform(X):
    return torch.clamp((X + 1.0) / 2.0, 0.0, 1.0)

# 普通采样
def generalized_steps(x, x_cond, seq, model, b, eta=0.):
    with torch.no_grad():
        n = x.size(0)       # 获取输入张量 x 的批次大小（样本数量）。
        # 这行代码用于生成下一个时间步序列seq_next, [-1]在列表的开头添加了一个 -1, list(seq[:-1])从seq列表中去掉最后一个时间步,得到一个新的列表,二者拼接得到seq_next.
        seq_next = [-1] + list(seq[:-1])
        x0_preds = []       # 初始化一个列表，用于存储中间结果 x0_t。
        xs = [x]            # 初始化一个列表，用于存储采样过程中生成的图像序列。将初始噪声图像 x 添加到 xs 中。
        # 这个循环用于在采样时间步骤中迭代.由于采样是从后往前进行的,所以使用 reversed() 将 seq 和 seq_next 反转后再进行迭代.
        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(n) * i).to(x.device)        # 创建一个张量 t,其中每个元素都为时间步 i.并将 t 移动到与输入张量 x 相同的设备上.
            next_t = (torch.ones(n) * j).to(x.device)   # 创建一个张量 next_t,其中每个元素都为下一个时间步 j.并将 next_t 移动到与输入张量 x 相同的设备上.
            at = compute_alpha(b, t.long())             # 调用 compute_alpha() 函数，计算时间步 t 对应的 alpha 值。
            at_next = compute_alpha(b, next_t.long())   # 调用 compute_alpha() 函数，计算下一个时间步 next_t 对应的 alpha 值。
            xt = xs[-1].to('cuda')                      # xs[-1] 表示当前时间步的图像，也就是模型在当前时间步生成的图像.将当前时间步的图像 xt 取为 xs 列表中最后一个图像，并将其移动到 CUDA 设备上(如果有可用的 CUDA 设备).xs[-1]的维度为[n, 3, self.config.data.image_size, self.config.data.image_size]

            et = model(torch.cat([x_cond, xt], dim=1), t)   # 通过模型 model 对图像进行采样，得到采样噪声 εθ(x~,x_t,t),其中x_cond和xt维度一样,按照dim=1拼接即可.
            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()  # 计算根据当前时间步加噪得到的图像x0_t。
            x0_preds.append(x0_t.to('cpu'))                 # 将 x0_t 添加到 x0_preds 列表中，并将其移动到 CPU 设备上。

            c1 = eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()   # 计算扩散步骤大小的调整参数 c1。
            c2 = ((1 - at_next) - c1 ** 2).sqrt()           # 计算扩散步骤大小的调整参数 c2。
            xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et    # at_next.sqrt() * x0_t + c2 * et即为确定性隐式抽样公式, 计算下一个时间步的图像 xt-1(xt_next)。
            xs.append(xt_next.to('cpu'))                    # 将 xt-1(xt_next) 添加到 xs 列表中，并将其移动到 CPU 设备上。
    return xs, x0_preds                                     # 最后，函数返回两个结果：xs 表示采样过程中生成的图像序列，x0_preds 表示采样过程中计算得到

# 分块采样
def generalized_steps_overlapping(x, x_cond, seq, model, b, eta=0., corners=None, p_size=None, manual_batching=True):
    with torch.no_grad():
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        x0_preds = []
        xs = [x]

        x_grid_mask = torch.zeros_like(x_cond, device=x.device)
        for (hi, wi) in corners:
            x_grid_mask[:, :, hi:hi + p_size, wi:wi + p_size] += 1

        # 这个循环用于在采样时间步骤中迭代.由于采样是从后往前进行的,所以使用 reversed() 将 seq 和 seq_next 反转后再进行迭代.
        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(n) * i).to(x.device)    # 创建一个张量 t,其中每个元素都为时间步 i.并将 t 移动到与输入张量 x 相同的设备上.
            next_t = (torch.ones(n) * j).to(x.device)   # 创建一个张量 next_t,其中每个元素都为下一个时间步 j.并将 next_t 移动到与输入张量 x 相同的设备上.
            at = compute_alpha(b, t.long())
            at_next = compute_alpha(b, next_t.long())
            xt = xs[-1].to('cuda')      # xs[-1] 表示当前时间步的图像，也就是模型在当前时间步生成的图像,x的值为与x_cond尺寸相同的高斯噪声.
            et_output = torch.zeros_like(x_cond, device=x.device)   # 用于存储采样后的输出。
            
            if manual_batching:
                manual_batching_size = 64
                # 将当前时间步的图像xt,有雾图像x_cond按照grid的尺寸和位置进行裁剪,并将裁剪后的图像按照dim=0拼接,形成xt_patch和x_cond_patch.
                xt_patch = torch.cat([crop(xt, hi, wi, p_size, p_size) for (hi, wi) in corners], dim=0)
                x_cond_patch = torch.cat([data_transform(crop(x_cond, hi, wi, p_size, p_size)) for (hi, wi) in corners], dim=0)
                # 对corners列表中的位置信息进行批处理.manual_batching_size 是指定的批处理大小,它决定了每次处理多少个位置的图像块.
                for i in range(0, len(corners), manual_batching_size):
                    outputs = model(torch.cat([x_cond_patch[i:i+manual_batching_size], 
                                               xt_patch[i:i+manual_batching_size]], dim=1), t)
                    for idx, (hi, wi) in enumerate(corners[i:i+manual_batching_size]):
                        et_output[0, :, hi:hi + p_size, wi:wi + p_size] += outputs[idx]
            else:
                for (hi, wi) in corners:
                    xt_patch = crop(xt, hi, wi, p_size, p_size)
                    x_cond_patch = crop(x_cond, hi, wi, p_size, p_size)
                    x_cond_patch = data_transform(x_cond_patch)
                    et_output[:, :, hi:hi + p_size, wi:wi + p_size] += model(torch.cat([x_cond_patch, xt_patch], dim=1), t)

            # 结合论文后的个人理解:不是没有进行融合操作,是直接把每次按照grid_r和patch划分出来的图像计算噪声估计后,就将计算后的值直接按照其在原图的位置加入到与有雾的原图大小一致的零矩阵中,且通过x_grid_mask中存储的每个元素的计算次数进行归一化,最终得到完成的平滑的原图噪声估计图.
            et = torch.div(et_output, x_grid_mask)      # 将每个grid_r的噪音除以重叠数目,以获得整体图片的预测噪音.
            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()  # 计算根据当前时间步加噪得到的图像x0_t.
            x0_preds.append(x0_t.to('cpu'))

            c1 = eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            c2 = ((1 - at_next) - c1 ** 2).sqrt()
            xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et
            xs.append(xt_next.to('cpu'))    # 将xt-1(xt_next)添加到 xs 列表中,并将其移动到 CPU 设备上.
    return xs, x0_preds
