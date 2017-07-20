import os
import numpy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F

from learning_args import parse_args
from base_demo import BaseDemo
from model import Net, GtNet
from visualizer import Visualizer
import logging
logging.basicConfig(format='[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s',
                            level=logging.INFO)


class Demo(BaseDemo):
    def __init__(self, args):
        super(Demo, self).__init__(args)
        self.model, self.model_gt = self.init_model(self.data.m_kernel)
        self.visualizer = Visualizer(args, self.data.reverse_m_dict)

    def init_model(self, m_kernel):
        self.model = Net(self.im_size, self.im_size, 3, self.num_frame - 1,
                             m_kernel.shape[1], self.m_range, m_kernel)
        self.model_gt = GtNet(self.im_size, self.im_size, 3, self.num_frame - 1,
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
            im_diff = (1 - appear).expand_as(im_output) * (im_pred - im_output)
            im_diff = im_diff / (1 - appear).sum(3).sum(2).expand_as(im_diff)
            loss = torch.abs(im_diff).sum() * im_diff.size(2) * im_diff.size(3)
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
            im_diff = (1 - appear).expand_as(im_output) * (im_pred - im_output)
            im_diff = im_diff / (1 - appear).sum(3).sum(2).expand_as(im_diff)
            loss = torch.abs(im_diff).sum() * im_diff.size(2) * im_diff.size(3)

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
            im_pred, m_mask, disappear, appear = self.model_gt(im_input, gt_motion)
            im_diff = (1 - appear).expand_as(im_output) * (im_pred - im_output)
            im_diff = im_diff / (1 - appear).sum(3).sum(2).expand_as(im_diff)
            loss = torch.abs(im_diff).sum() * im_diff.size(2) * im_diff.size(3)

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


def main():
    args = parse_args()
    logging.info(args)
    demo = Demo(args)
    if args.train:
        demo.train_unsupervised()
    if args.test:
        demo.test_unsupervised()
    if args.test_gt:
        demo.test_gt_unsupervised()

if __name__ == '__main__':
    main()
