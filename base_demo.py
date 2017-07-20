import os
import numpy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F

from learning_args import parse_args
from data.mnist_data import MnistData
from data.box_data import BoxData
from data.box_mnist_data import BoxMnistData
from base_model import BaseNet, BaseGtNet
from visualize.base_visualizer import BaseVisualizer
import logging
logging.basicConfig(format='[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s',
                            level=logging.INFO)


class BaseDemo(object):
    def __init__(self, args):
        self.learning_rate = args.learning_rate
        self.train_epoch = args.train_epoch
        self.test_epoch = args.test_epoch
        self.test_interval = args.test_interval
        self.save_dir = args.save_dir
        self.display = args.display
        self.best_improve_percent = -1e10
        self.batch_size = args.batch_size
        self.im_size = args.image_size
        self.num_frame = args.num_frame
        self.m_range = args.motion_range
        if args.data == 'box':
            self.data = BoxData(args)
        elif args.data == 'mnist':
            self.data = MnistData(args)
        elif args.data == 'box_mnist':
            self.data = BoxMnistData(args)
        self.init_model_path = args.init_model_path
        self.model, self.model_gt = self.init_model(self.data.m_kernel)
        self.visualizer = BaseVisualizer(args, self.data.reverse_m_dict)

    def init_model(self, m_kernel):
        self.model = BaseNet(self.im_size, self.im_size, 3, self.num_frame - 1,
                             m_kernel.shape[1], self.m_range, m_kernel)
        self.model_gt = BaseGtNet(self.im_size, self.im_size, 3, self.num_frame - 1,
                                  m_kernel.shape[1], self.m_range, m_kernel)
        if torch.cuda.is_available():
            # model = torch.nn.DataParallel(model).cuda()
            self.model = self.model.cuda()
            self.model_gt = self.model_gt.cuda()
        if self.init_model_path is not '':
            self.model.load_state_dict(torch.load(self.init_model_path))
        return self.model, self.model_gt

    def train_unsupervised(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        base_loss, train_loss = [], []
        for epoch in range(self.train_epoch):
            optimizer.zero_grad()
            im, _, _ = self.data.get_next_batch(self.data.train_images)
            im_input = im[:, :-1, :, :, :].reshape(self.batch_size, -1, self.im_size, self.im_size)
            im_output = im[:, -1, :, :, :]
            im_input = Variable(torch.from_numpy(im_input).float())
            im_output = Variable(torch.from_numpy(im_output).float())
            if torch.cuda.is_available():
                im_input, im_output = im_input.cuda(), im_output.cuda()
            im_pred, m_mask, disappear, appear = self.model(im_input)
            loss = torch.abs(im_pred - im_output).sum()
            loss.backward()
            optimizer.step()

            train_loss.append(loss.data[0])
            if len(train_loss) > 100:
                train_loss.pop(0)
            ave_train_loss = sum(train_loss) / float(len(train_loss))
            base_loss.append(torch.abs(im_input[:, -3:, :, :] - im_output).sum().data[0])
            if len(base_loss) > 100:
                base_loss.pop(0)
            ave_base_loss = sum(base_loss) / float(len(base_loss))
            logging.info('epoch %d, train loss: %.2f, average train loss: %.2f, base loss: %.2f',
                         epoch, loss.data[0], ave_train_loss, ave_base_loss)
            if (epoch+1) % self.test_interval == 0:
                logging.info('epoch %d, testing', epoch)
                self.validate()

    def validate(self):
        improve_percent = self.test_unsupervised()
        if improve_percent >= self.best_improve_percent:
            logging.info('model save to %s', os.path.join(self.save_dir, 'model.pth'))
            with open(os.path.join(self.save_dir, 'model.pth'), 'w') as handle:
                torch.save(self.model.state_dict(), handle)
            self.best_improve_percent = improve_percent
        logging.info('current best improved percent: %.2f', self.best_improve_percent)

    def test_unsupervised(self):
        base_loss, test_loss = [], []
        test_epe = []
        for epoch in range(self.test_epoch):
            im, motion, _ = self.data.get_next_batch(self.data.test_images)
            im_input = im[:, :-1, :, :, :].reshape(self.batch_size, -1, self.im_size, self.im_size)
            im_output = im[:, -1, :, :, :]
            gt_motion = motion[:, -2, :, :, :]
            im_input = Variable(torch.from_numpy(im_input).float())
            im_output = Variable(torch.from_numpy(im_output).float())
            gt_motion = Variable(torch.from_numpy(gt_motion).float())
            if torch.cuda.is_available():
                im_input, im_output = im_input.cuda(), im_output.cuda()
                gt_motion = gt_motion.cuda()
            im_pred, m_mask, disappear, appear = self.model(im_input)
            loss = torch.abs(im_pred - im_output).sum()

            test_loss.append(loss.data[0])
            base_loss.append(torch.abs(im_input[:, -3:, :, :] - im_output).sum().data[0])
            flow = self.motion2flow(m_mask)
            epe = (flow - gt_motion) * (flow - gt_motion)
            epe = torch.sqrt(epe.sum(1))
            epe = epe.sum() / epe.numel()
            test_epe.append(epe.cpu().data[0])
            if self.display:
                self.visualizer.visualize_result(im_input, im_output, im_pred, flow, gt_motion,
                                                 disappear, appear, 'test_%d.png' % epoch)
        test_loss = numpy.mean(numpy.asarray(test_loss))
        base_loss = numpy.mean(numpy.asarray(base_loss))
        improve_loss = base_loss - test_loss
        improve_percent = improve_loss / (base_loss + 1e-5)
        logging.info('average test loss: %.2f, base loss: %.2f', test_loss, base_loss)
        logging.info('improve_loss: %.2f, improve_percent: %.2f', improve_loss, improve_percent)
        test_epe = numpy.mean(numpy.asarray(test_epe))
        logging.info('average test endpoint error: %.2f', test_epe)
        return improve_percent

    def test_gt_unsupervised(self):
        base_loss, test_loss = [], []
        test_epe = []
        for epoch in range(self.test_epoch):
            im, motion, depth = self.data.get_next_batch(self.data.test_images)
            im_input = im[:, :-1, :, :, :].reshape(self.batch_size, -1, self.im_size, self.im_size)
            im_output = im[:, -1, :, :, :]
            gt_motion = motion[:, -2, :, :, :]
            gt_depth = depth[:, -2, :, :, :]
            im_input = Variable(torch.from_numpy(im_input).float())
            im_output = Variable(torch.from_numpy(im_output).float())
            gt_motion = Variable(torch.from_numpy(gt_motion).float())
            gt_depth = Variable(torch.from_numpy(gt_depth))
            if torch.cuda.is_available():
                im_input, im_output = im_input.cuda(), im_output.cuda()
                gt_motion = gt_motion.cuda()
                gt_depth = gt_depth.cuda()
            im_pred, m_mask, disappear, appear = self.model_gt(im_input, gt_motion)
            loss = torch.abs(im_pred - im_output).sum()

            test_loss.append(loss.data[0])
            base_loss.append(torch.abs(im_input[:, -3:, :, :] - im_output).sum().data[0])
            flow = self.motion2flow(m_mask)
            epe = (flow - gt_motion) * (flow - gt_motion)
            epe = torch.sqrt(epe.sum(1))
            epe = epe.sum() / epe.numel()
            test_epe.append(epe.cpu().data[0])
            if self.display:
                self.visualizer.visualize_result(im_input, im_output, im_pred, flow, gt_motion,
                                                 disappear, appear, 'test_gt.png')
        test_loss = numpy.mean(numpy.asarray(test_loss))
        base_loss = numpy.mean(numpy.asarray(base_loss))
        improve_loss = base_loss - test_loss
        improve_percent = improve_loss / (base_loss + 1e-5)
        logging.info('average ground truth test loss: %.2f, base loss: %.2f', test_loss, base_loss)
        logging.info('improve_loss: %.2f, improve_percent: %.2f', improve_loss, improve_percent)
        test_epe = numpy.mean(numpy.asarray(test_epe))
        logging.info('average ground truth test endpoint error: %.2f', test_epe)
        return improve_percent

    def motion2flow(self, m_mask):
        reverse_m_dict = self.data.reverse_m_dict
        [batch_size, num_class, height, width] = m_mask.size()
        kernel_x = Variable(torch.zeros(batch_size, num_class, height, width))
        kernel_y = Variable(torch.zeros(batch_size, num_class, height, width))
        if torch.cuda.is_available():
            kernel_x = kernel_x.cuda()
            kernel_y = kernel_y.cuda()
        for i in range(num_class):
            (m_x, m_y) = reverse_m_dict[i]
            kernel_x[:, i, :, :] = m_x
            kernel_y[:, i, :, :] = m_y
        flow = Variable(torch.zeros(batch_size, 2, height, width))
        if torch.cuda.is_available():
            flow = flow.cuda()
        flow[:, 0, :, :] = (m_mask * kernel_x).sum(1)
        flow[:, 1, :, :] = (m_mask * kernel_y).sum(1)
        return flow


def main():
    args = parse_args()
    logging.info(args)
    demo = BaseDemo(args)
    if args.train:
        demo.train_unsupervised()
    if args.test:
        demo.test_unsupervised()
    if args.test_gt:
        demo.test_gt_unsupervised()

if __name__ == '__main__':
    main()
