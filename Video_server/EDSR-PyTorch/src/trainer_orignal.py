import os
import math
import imageio
from decimal import Decimal
import time

import utility
import torch
import torch.nn.utils as utils
from tqdm import tqdm
import torch, gc###

class Trainer():
    def __init__(self, args, loader, my_model, my_loss, ckp):
        self.args = args
        self.scale = args.scale
        self.skip = args.skip
        self.ckp = ckp
        self.loader_train = loader.loader_train#data.Data
        self.loader_test = loader.loader_test

        self.model = my_model
        self.loss = my_loss
        #待训练参数
        '''
                      'model.HFABs.0.bn1.weight', 'model.HFABs.0.bn1.bias', 'model.HFABs.0.bn2.weight',
                      'model.HFABs.0.bn2.bias', 'model.HFABs.0.bn3.weight', 'model.HFABs.0.bn3.bias',
                      'model.HFABs.0.squeeze.weight',
                      'model.HFABs.0.squeeze.bias', 'model.HFABs.0.convs.0.conv1.expand_conv.weight',
                      'model.HFABs.0.convs.0.conv1.expand_conv.bias',
                      'model.HFABs.0.convs.0.conv1.fea_conv.weight', 'model.HFABs.0.convs.0.conv1.fea_conv.bias',
                      'model.HFABs.0.convs.0.conv1.reduce_conv.weight',
                      'model.HFABs.0.convs.0.conv1.reduce_conv.bias', 'model.HFABs.0.convs.0.conv2.expand_conv.weight',
                      'model.HFABs.0.convs.0.conv2.expand_conv.bias',
                      'model.HFABs.0.convs.0.conv2.fea_conv.weight', 'model.HFABs.0.convs.0.conv2.fea_conv.bias',
                      'model.HFABs.0.convs.0.conv2.reduce_conv.weight', 'model.HFABs.0.convs.0.conv2.reduce_conv.bias',
                      'model.HFABs.0.excitate.weight', 'model.HFABs.0.excitate.bias',

                      'model.HFABs.1.bn1.weight', 'model.HFABs.1.bn1.bias', 'model.HFABs.1.bn2.weight',
                      'model.HFABs.1.bn2.bias',
                      'model.HFABs.1.bn3.weight', 'model.HFABs.1.bn3.bias', 'model.HFABs.1.squeeze.weight',
                      'model.HFABs.1.squeeze.bias',
                      'model.HFABs.1.convs.0.conv1.expand_conv.weight', 'model.HFABs.1.convs.0.conv1.expand_conv.bias',
                      'model.HFABs.1.convs.0.conv1.fea_conv.weight',
                      'model.HFABs.1.convs.0.conv1.fea_conv.bias', 'model.HFABs.1.convs.0.conv1.reduce_conv.weight',
                      'model.HFABs.1.convs.0.conv1.reduce_conv.bias',
                      'model.HFABs.1.convs.0.conv2.expand_conv.weight', 'model.HFABs.1.convs.0.conv2.expand_conv.bias',
                      'model.HFABs.1.convs.0.conv2.fea_conv.weight',
                      'model.HFABs.1.convs.0.conv2.fea_conv.bias', 'model.HFABs.1.convs.0.conv2.reduce_conv.weight',
                      'model.HFABs.1.convs.0.conv2.reduce_conv.bias', 'model.HFABs.1.excitate.weight',
                      'model.HFABs.1.excitate.bias',

                      'model.HFABs.2.bn1.weight', 'model.HFABs.2.bn1.bias', 'model.HFABs.2.bn2.weight',
                      'model.HFABs.2.bn2.bias',
                      'model.HFABs.2.bn3.weight', 'model.HFABs.2.bn3.bias', 'model.HFABs.2.squeeze.weight',
                      'model.HFABs.2.squeeze.bias', 'model.HFABs.2.convs.0.conv1.expand_conv.weight',
                      'model.HFABs.2.convs.0.conv1.expand_conv.bias',
                      'model.HFABs.2.convs.0.conv1.fea_conv.weight', 'model.HFABs.2.convs.0.conv1.fea_conv.bias',
                      'model.HFABs.2.convs.0.conv1.reduce_conv.weight', 'model.HFABs.2.convs.0.conv1.reduce_conv.bias',
                      'model.HFABs.2.convs.0.conv2.expand_conv.weight', 'model.HFABs.2.convs.0.conv2.expand_conv.bias',
                      'model.HFABs.2.convs.0.conv2.fea_conv.weight', 'model.HFABs.2.convs.0.conv2.fea_conv.bias',
                      'model.HFABs.2.convs.0.conv2.reduce_conv.weight', 'model.HFABs.2.convs.0.conv2.reduce_conv.bias',
                      'model.HFABs.2.excitate.weight', 'model.HFABs.2.excitate.bias',

                      'model.HFABs.3.bn1.weight', 'model.HFABs.3.bn1.bias', 'model.HFABs.3.bn2.weight',
                      'model.HFABs.3.bn2.bias',
                      'model.HFABs.3.bn3.weight', 'model.HFABs.3.bn3.bias', 'model.HFABs.3.squeeze.weight',
                      'model.HFABs.3.squeeze.bias', 'model.HFABs.3.convs.0.conv1.expand_conv.weight',
                      'model.HFABs.3.convs.0.conv1.expand_conv.bias',
                      'model.HFABs.3.convs.0.conv1.fea_conv.weight', 'model.HFABs.3.convs.0.conv1.fea_conv.bias',
                      'model.HFABs.3.convs.0.conv1.reduce_conv.weight',
                      'model.HFABs.3.convs.0.conv1.reduce_conv.bias', 'model.HFABs.3.convs.0.conv2.expand_conv.weight',
                      'model.HFABs.3.convs.0.conv2.expand_conv.bias',
                      'model.HFABs.3.convs.0.conv2.fea_conv.weight', 'model.HFABs.3.convs.0.conv2.fea_conv.bias',
                      'model.HFABs.3.convs.0.conv2.reduce_conv.weight',
                      'model.HFABs.3.convs.0.conv2.reduce_conv.bias', 'model.HFABs.3.excitate.weight',
                      'model.HFABs.3.excitate.bias',

                      'model.lr_conv.weight', 'model.lr_conv.bias',

        '''
        #'''冻结
        para_name = [ 'model.head.weight', 'model.head.bias',

                      'model.warmup.0.weight', 'model.warmup.0.bias',
                      'model.warmup.1.bn1.weight', 'model.warmup.1.bn1.bias', 'model.warmup.1.bn2.weight',
                      'model.warmup.1.bn2.bias', 'model.warmup.1.bn3.weight', 'model.warmup.1.bn3.bias',
                      'model.warmup.1.squeeze.weight', 'model.warmup.1.squeeze.bias','model.warmup.1.convs.0.conv1.expand_conv.weight',
                      'model.warmup.1.convs.0.conv1.expand_conv.bias', 'model.warmup.1.convs.0.conv1.fea_conv.weight',
                      'model.warmup.1.convs.0.conv1.fea_conv.bias', 'model.warmup.1.convs.0.conv1.reduce_conv.weight',
                      'model.warmup.1.convs.0.conv1.reduce_conv.bias', 'model.warmup.1.convs.0.conv2.expand_conv.weight',
                      'model.warmup.1.convs.0.conv2.expand_conv.bias', 'model.warmup.1.convs.0.conv2.fea_conv.weight',
                      'model.warmup.1.convs.0.conv2.fea_conv.bias', 'model.warmup.1.convs.0.conv2.reduce_conv.weight',
                      'model.warmup.1.convs.0.conv2.reduce_conv.bias', 'model.warmup.1.convs.1.conv1.expand_conv.weight',
                      'model.warmup.1.convs.1.conv1.expand_conv.bias', 'model.warmup.1.convs.1.conv1.fea_conv.weight',
                      'model.warmup.1.convs.1.conv1.fea_conv.bias', 'model.warmup.1.convs.1.conv1.reduce_conv.weight',
                      'model.warmup.1.convs.1.conv1.reduce_conv.bias', 'model.warmup.1.convs.1.conv2.expand_conv.weight',
                      'model.warmup.1.convs.1.conv2.expand_conv.bias', 'model.warmup.1.convs.1.conv2.fea_conv.weight',
                      'model.warmup.1.convs.1.conv2.fea_conv.bias', 'model.warmup.1.convs.1.conv2.reduce_conv.weight',
                      'model.warmup.1.convs.1.conv2.reduce_conv.bias', 'model.warmup.1.excitate.weight', 'model.warmup.1.excitate.bias',


                      'model.ERBs.0.conv1.expand_conv.weight','model.ERBs.0.conv1.expand_conv.bias','model.ERBs.0.conv1.fea_conv.weight',
                      'model.ERBs.0.conv1.fea_conv.bias','model.ERBs.0.conv1.reduce_conv.weight','model.ERBs.0.conv1.reduce_conv.bias',
                      'model.ERBs.0.conv2.expand_conv.weight','model.ERBs.0.conv2.expand_conv.bias','model.ERBs.0.conv2.fea_conv.weight',
                      'model.ERBs.0.conv2.fea_conv.bias','model.ERBs.0.conv2.reduce_conv.weight','model.ERBs.0.conv2.reduce_conv.bias',

                      'model.ERBs.1.conv1.expand_conv.weight','model.ERBs.1.conv1.expand_conv.bias','model.ERBs.1.conv1.fea_conv.weight',
                      'model.ERBs.1.conv1.fea_conv.bias','model.ERBs.1.conv1.reduce_conv.weight','model.ERBs.1.conv1.reduce_conv.bias',
                      'model.ERBs.1.conv2.expand_conv.weight', 'model.ERBs.1.conv2.expand_conv.bias','model.ERBs.1.conv2.fea_conv.weight',
                      'model.ERBs.1.conv2.fea_conv.bias','model.ERBs.1.conv2.reduce_conv.weight','model.ERBs.1.conv2.reduce_conv.bias',

                      'model.ERBs.2.conv1.expand_conv.weight', 'model.ERBs.2.conv1.expand_conv.bias',
                      'model.ERBs.2.conv1.fea_conv.weight','model.ERBs.2.conv1.fea_conv.bias',
                      'model.ERBs.2.conv1.reduce_conv.weight',
                      'model.ERBs.2.conv1.reduce_conv.bias', 'model.ERBs.2.conv2.expand_conv.weight',
                      'model.ERBs.2.conv2.expand_conv.bias',
                      'model.ERBs.2.conv2.fea_conv.weight', 'model.ERBs.2.conv2.fea_conv.bias',
                      'model.ERBs.2.conv2.reduce_conv.weight',
                      'model.ERBs.2.conv2.reduce_conv.bias',

                      'model.ERBs.3.conv1.expand_conv.weight', 'model.ERBs.3.conv1.expand_conv.bias',
                      'model.ERBs.3.conv1.fea_conv.weight', 'model.ERBs.3.conv1.fea_conv.bias',
                      'model.ERBs.3.conv1.reduce_conv.weight',
                      'model.ERBs.3.conv1.reduce_conv.bias', 'model.ERBs.3.conv2.expand_conv.weight',
                      'model.ERBs.3.conv2.expand_conv.bias',
                      'model.ERBs.3.conv2.fea_conv.weight', 'model.ERBs.3.conv2.fea_conv.bias',
                      'model.ERBs.3.conv2.reduce_conv.weight',
                      'model.ERBs.3.conv2.reduce_conv.bias',



                      'model.tail.0.weight', 'model.tail.0.bias'

                      ]#
        for name, parameter in self.model.named_parameters():
            # self.ckp.write_log("'{}',".format(name))
            if name in para_name:
                # self.ckp.write_log("{}".format(parameter))
                parameter.requires_grad = False
                #self.ckp.write_log("{}".format(str(i)))
            self.ckp.write_log('{}: {}'.format(name, parameter.requires_grad))
        #'''
        #优化器
        self.optimizer = utility.make_optimizer(args, self.model)#############

        if self.args.load != '':
            self.optimizer.load(ckp.dir, epoch=len(ckp.log))#ckp.dir='/home/srteam/lrq/EDSR-PyTorch', 'experiment', args.load

        self.error_last = 1e8

    def train(self):
        t0 = time.time()#######
        self.loss.step()######lr_scheduler.step
        epoch = self.optimizer.get_last_epoch() + 1
        lr = self.optimizer.get_lr()

        self.ckp.write_log(
            '[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr))
        )
        self.loss.start_log()
        self.model.train()

        timer_data, timer_model = utility.timer(), utility.timer()
        # TEMP
        self.loader_train.dataset.set_scale(0)##
        for batch, (lr, hr, _,) in enumerate(self.loader_train):#[6, 3, 192, 192]
            lr, hr = self.prepare(lr, hr)
            timer_data.hold()
            timer_model.tic()

            self.optimizer.zero_grad()
            sr = self.model(lr, 0)
            loss = self.loss(sr, hr)##############
            loss.backward()
            if self.args.gclip > 0:
                utils.clip_grad_value_(
                    self.model.parameters(),
                    self.args.gclip
                )
            self.optimizer.step()#更新参数
            timer_model.hold()

            if (batch + 1) % self.args.print_every == 0:##################
                self.ckp.write_log('[{}/{}]\t{}\t{:.1f}+{:.1f}s'.format(
                    (batch + 1) * self.args.batch_size,
                    len(self.loader_train.dataset),
                    self.loss.display_loss(batch),
                    timer_model.release(),
                    timer_data.release()))
            

            timer_data.tic()
        t1 = time.time()#######
        train_t=t1-t0
        self.ckp.write_log('Train time：{:.3f}s\n'.format(train_t))######

        self.loss.end_log(len(self.loader_train))
        self.error_last = self.loss.log[-1, -1]
        self.optimizer.schedule()##########

    def test(self):
        torch.set_grad_enabled(False)

        epoch = self.optimizer.get_last_epoch()
        self.ckp.write_log('\nEvaluation:')
        self.ckp.add_log(
            torch.zeros(1, len(self.loader_test), len(self.scale))
        )

        self.model.eval()

        timer_test = utility.timer()
        diffs=0
        path = r'/home/srteam/lrq/EDSR-PyTorch/experiment/{}/test{}.txt'.format(self.args.data_test[0], self.scale[0])
        if self.args.save_results: self.ckp.begin_background()######
        for idx_data, d in enumerate(self.loader_test):
            for idx_scale, scale in enumerate(self.scale):
                d.dataset.set_scale(idx_scale)
                for lr, hr, filename in tqdm(d, ncols=80):
                    print(hr)
                    lr, hr = self.prepare(lr, hr)
                    print(hr)
                    t0 = time.time()#######
                    sr = self.model(lr, idx_scale)
                    diff = time.time() - t0##########
                    ##gc.collect()  #####
                    ##torch.cuda.empty_cache()  ###
                    self.ckp.write_log('\nSR_one: {:.3f}s\n'.format(diff))
                    diffs += diff
                    self.ckp.write_log('SR: {:.3f}s\n'.format(diffs))
                    #with open(path, "a") as f:
                    #    f.write('{:.3f}\n'.format(diff))

        #'''
                    sr = utility.quantize(sr, self.args.rgb_range)

                    save_list = [sr]
                    self.ckp.log[-1, idx_data, idx_scale] += utility.calc_psnr(sr, hr, scale, self.args.rgb_range, dataset=d)#################################
                    if self.args.save_gt:
                        save_list.extend([lr, hr])

                    if self.args.save_results:##########
                        self.ckp.save_results(d, filename[0], save_list, scale)


                self.ckp.log[-1, idx_data, idx_scale] /= len(d)
                best = self.ckp.log.max(0)##########pnse max
                self.ckp.write_log(
                    '[{} x{}]\tPSNR: {:.3f} (Best: {:.3f} @epoch {})'.format(
                        d.dataset.name,
                        scale,
                        self.ckp.log[-1, idx_data, idx_scale],
                        best[0][idx_data, idx_scale],
                        best[1][idx_data, idx_scale] + 1
                    )
                )

        self.ckp.write_log('Forward: {:.2f}s\n'.format(timer_test.toc()))
        self.ckp.write_log('Saving...')

        if self.args.save_results:
            self.ckp.end_background()

        if not self.args.test_only:
            self.ckp.save(self, epoch, is_best=(best[1][0, 0] + 1 == epoch))###########
        
        self.ckp.write_log('Total: {:.2f}s\n'.format(timer_test.toc()), refresh=True)
        #'''
        torch.set_grad_enabled(True)

    def prepare(self, *args):
        if self.args.cpu:
            device = torch.device('cpu')
        else:
            if torch.backends.mps.is_available():
                device = torch.device('mps')
            elif torch.cuda.is_available():
                device = torch.device('cuda')
            else:
                device = torch.device('cpu')
        def _prepare(tensor):
            if self.args.precision == 'half': tensor = tensor.half()
            return tensor.to(device)

        return [_prepare(a) for a in args]

    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            epoch = self.optimizer.get_last_epoch() + 1
            return epoch >= self.args.epochs

