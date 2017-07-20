import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F


class BaseNet(nn.Module):
    def __init__(self, im_height, im_width, im_channel, n_inputs, n_class, m_range, m_kernel):
        super(BaseNet, self).__init__()
        num_hidden = 64
        self.conv0 = nn.Conv2d(n_inputs*im_channel, num_hidden, 3, 1, 1)
        self.bn0 = nn.BatchNorm2d(num_hidden)
        self.conv1 = nn.Conv2d(num_hidden, num_hidden, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(num_hidden)
        self.conv2 = nn.Conv2d(num_hidden, num_hidden, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(num_hidden)
        self.conv3 = nn.Conv2d(num_hidden, num_hidden, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(num_hidden)
        self.conv4 = nn.Conv2d(num_hidden, num_hidden, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(num_hidden)
        self.conv5 = nn.Conv2d(num_hidden, num_hidden, 3, 1, 1)
        self.bn5 = nn.BatchNorm2d(num_hidden)
        self.conv6 = nn.Conv2d(num_hidden, num_hidden, 3, 1, 1)
        self.bn6 = nn.BatchNorm2d(num_hidden)
        self.conv7 = nn.Conv2d(num_hidden*2, num_hidden, 3, 1, 1)
        self.bn7 = nn.BatchNorm2d(num_hidden)
        self.conv8 = nn.Conv2d(num_hidden*2, num_hidden, 3, 1, 1)
        self.bn8 = nn.BatchNorm2d(num_hidden)
        self.conv9 = nn.Conv2d(num_hidden*2, num_hidden, 3, 1, 1)
        self.bn9 = nn.BatchNorm2d(num_hidden)
        self.conv10 = nn.Conv2d(num_hidden*2, num_hidden, 3, 1, 1)
        self.bn10 = nn.BatchNorm2d(num_hidden)
        self.conv11 = nn.Conv2d(num_hidden*2, num_hidden, 3, 1, 1)
        self.bn11 = nn.BatchNorm2d(num_hidden)
        self.conv = nn.Conv2d(num_hidden, n_class, 3, 1, 1)

        self.maxpool = nn.MaxPool2d(2, stride=2, return_indices=False, ceil_mode=False)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)

        self.im_height = im_height
        self.im_width = im_width
        self.im_channel = im_channel
        self.n_inputs = n_inputs
        self.n_class = n_class
        self.m_range = m_range
        m_kernel = m_kernel.swapaxes(0, 1)
        self.m_kernel = Variable(torch.from_numpy(m_kernel).float())
        if torch.cuda.is_available():
            self.m_kernel = self.m_kernel.cuda()

    def forward(self, im_input):
        x = self.bn0(self.conv0(im_input))
        x1 = F.relu(self.bn1(self.conv1(x)))
        x2 = self.maxpool(x1)
        x2 = F.relu(self.bn2(self.conv2(x2)))
        x3 = self.maxpool(x2)
        x3 = F.relu(self.bn3(self.conv3(x3)))
        x4 = self.maxpool(x3)
        x4 = F.relu(self.bn4(self.conv4(x4)))
        x5 = self.maxpool(x4)
        x5 = F.relu(self.bn5(self.conv5(x5)))
        x6 = self.maxpool(x5)
        x6 = F.relu(self.bn6(self.conv6(x6)))
        x6 = self.upsample(x6)
        x7 = torch.cat((x6, x5), 1)
        x7 = F.relu(self.bn7(self.conv7(x7)))
        x7 = self.upsample(x7)
        x8 = torch.cat((x7, x4), 1)
        x8 = F.relu(self.bn8(self.conv8(x8)))
        x8 = self.upsample(x8)
        x9 = torch.cat((x8, x3), 1)
        x9 = F.relu(self.bn9(self.conv9(x9)))
        x9 = self.upsample(x9)
        x10 = torch.cat((x9, x2), 1)
        x10 = F.relu(self.bn10(self.conv10(x10)))
        x10 = self.upsample(x10)
        x11 = torch.cat((x10, x1), 1)
        x11 = F.relu(self.bn11(self.conv11(x11)))
        m_mask = F.softmax(self.conv(x11))

        out_mask = F.conv2d(m_mask, self.m_kernel, None, 1, self.m_range, 1, self.m_kernel.size(0))
        seg = construct_seg(out_mask, self.m_kernel, self.m_range)
        appear = F.relu(1 - seg)
        disappear = F.relu(seg - 1)
        pred = construct_image(im_input[:, -self.im_channel:, :, :], out_mask, disappear, self.m_kernel, self.m_range)
        return pred, m_mask, disappear, appear


class BaseGtNet(nn.Module):
    def __init__(self, im_height, im_width, im_channel, n_inputs, n_class, m_range, m_kernel):
        super(BaseGtNet, self).__init__()
        self.im_height = im_height
        self.im_width = im_width
        self.im_channel = im_channel
        self.n_class = n_class
        self.m_range = m_range
        m_kernel = m_kernel.swapaxes(0, 1)
        self.m_kernel = Variable(torch.from_numpy(m_kernel).float())
        if torch.cuda.is_available():
            self.m_kernel = self.m_kernel.cuda()

    def forward(self, im_input, gt_motion):
        m_mask = self.motion2mask(gt_motion, self.n_class, self.m_range)
        out_mask = F.conv2d(m_mask, self.m_kernel, None, 1, self.m_range, 1, self.m_kernel.size(0))
        seg = construct_seg(out_mask, self.m_kernel, self.m_range)
        appear = F.relu(1 - seg)
        disappear = F.relu(seg - 1)
        pred = construct_image(im_input[:, -self.im_channel:, :, :], out_mask, disappear, self.m_kernel, self.m_range)
        return pred, m_mask, disappear, appear

    def motion2mask(self, motion, n_class, m_range):
        m_mask = Variable(torch.Tensor(motion.size(0), n_class, motion.size(2), motion.size(3)))
        if torch.cuda.is_available():
            m_mask = m_mask.cuda()
        motion_floor = torch.floor(motion.cpu().data).long()
        for i in range(motion.size(0)):
            for j in range(motion.size(2)):
                for k in range(motion.size(3)):
                    a = Variable(torch.zeros(int(math.sqrt(n_class))))
                    b = Variable(torch.zeros(int(math.sqrt(n_class))))
                    idx = motion_floor[i, 0, j, k] + m_range
                    a[idx] = 1 - (motion[i, 0, j, k] - motion_floor[i, 0, j, k])
                    a[idx + 1] = 1 - a[idx]
                    idx = motion_floor[i, 1, j, k] + m_range
                    b[idx] = 1 - (motion[i, 1, j, k] - motion_floor[i, 1, j, k])
                    b[idx + 1] = 1 - b[idx]
                    tmp = torch.ger(b, a)
                    m_mask[i, :, j, k] = tmp.view(-1)
        return m_mask


def construct_seg(out_mask, m_kernel, m_range):
    seg_expand = Variable(torch.ones(out_mask.size()))
    if torch.cuda.is_available():
        seg_expand = seg_expand.cuda()
    nearby_seg = F.conv2d(seg_expand, m_kernel, None, 1, m_range, 1, m_kernel.size(0))
    seg = (nearby_seg * out_mask).sum(1)
    return seg


def construct_image(im, out_mask, disappear, m_kernel, m_range):
    im = im * (1 - disappear).expand_as(im)
    pred = Variable(torch.Tensor(im.size()))
    if torch.cuda.is_available():
        pred = pred.cuda()
    for i in range(im.size(1)):
        im_expand = im[:, i, :, :].unsqueeze(1).expand_as(out_mask)
        nearby_im = F.conv2d(im_expand, m_kernel, None, 1, m_range, 1, m_kernel.size(0))
        pred[:, i, :, :] = (nearby_im * out_mask).sum(1)
    return pred

