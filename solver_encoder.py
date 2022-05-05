import os
import pickle
from model_vc import Generator
import torch
import numpy as np
import time
import matplotlib.pyplot as plt
import datetime
import soundfile as sf
from torch.utils.tensorboard import SummaryWriter
from vc_validation import validation
from torch.nn.utils.rnn import pad_sequence


class Solver(object):

    def __init__(self, vcc_loader, validation_loader, config):
        """Initialize configurations."""

        # Data loader.
        self.vcc_loader = vcc_loader
        self.validation_loader = validation_loader
        self.training_batch_size = config['training_batch_size']
        self.validation_batch_size = config['validation_batch_size']

        # task
        self.task = config['task']
        self.use_prior_flow = config['use_prior_flow']
        self.use_post_flow = config['use_post_flow']
        self.train_flag = False

        # continue to train or not
        self.continue_to_train = config['continue_to_train']
        self.iter = 0

        # Training configurations.
        self.num_iters = config['num_iter']
        
        # Miscellaneous.
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device(config['device'] if self.use_cuda else 'cpu')
        self.log_step = config['log_step']

        # Build the model and tensorboard.
        self.build_model(config)
        if not os.path.exists(os.path.join(config['log_event'], self.task)):
            os.mkdir(os.path.join(config['log_event'], self.task))
        self.writer = SummaryWriter(os.path.join(config['log_event'], self.task))
        if not os.path.exists(os.path.join(config['log_model'], self.task)):
            os.mkdir(os.path.join(config['log_model'], self.task))
        self.model_save = os.path.join(config['log_model'], self.task)

        # whether to begin to train post_flow
        self.begin_post_flow = config['post_glow_training_start']

        # validation
        self.conf = config
        if config['is_training']:
            self.validate = validation()
            for i in range(1, 5, 1):
                wav = 'gt_audio' + str(i)
                self.writer.add_figure('gt_mel/'+self.conf[wav][5:], self.plot_spectrogram(
                    np.load(os.path.join(config['mel_folder'], config[wav]+'.npy'))))
                self.writer.add_audio('gt_audio/'+self.conf[wav][5:], torch.tensor(sf.read(
                    os.path.join(config['gt_audio'], config[wav]+'.wav'))[0]).to(self.device), sample_rate=16000)

    def build_model(self, config):
        self.G = Generator(config)
        if self.continue_to_train is True:
            checkpoint = torch.load(config['load_model'], map_location={'cuda:1': 'cuda:0'})
            self.G.load_state_dict(checkpoint)
            self.iter = config['iter'] + 1
        self.G.to(self.device)
        self.g_optimizer1 = torch.optim.Adam(self.G.parameters(), lr=0.0002, betas=(0.9, 0.98), eps=1e-08, weight_decay=0)
        self.g_optimizer2 = torch.optim.Adam(self.G.post_flow.parameters(), lr=0.001, betas=(0.9, 0.98),
                                             eps=1e-08, weight_decay=0)
        self.g_optimizer = self.g_optimizer1

    def reset_grad(self):
        """Reset the gradient buffers."""
        self.g_optimizer.zero_grad()

    def mse_loss(self, input_a, input_b, nonpadding):
        return torch.square(input_a - input_b).sum() / (80 * sum(nonpadding[nonpadding == 1]))

    def l1_loss(self, input_a, input_b, nonpadding):
        return torch.abs_(input_a - input_b).sum() / (80 * sum(nonpadding[nonpadding == 1]))

    def plot_spectrogram(self, spectrogram):
        if len(spectrogram.shape) == 3:
            spectrogram = spectrogram.squeeze(0)    # discard the batch dim
        if not spectrogram.shape[0] == 80:
            spectrogram = spectrogram.T
        fig, ax = plt.subplots(figsize=(10, 2))
        im = ax.imshow(spectrogram, aspect="auto", origin="lower",
                       interpolation='none')
        plt.colorbar(im, ax=ax)

        fig.canvas.draw()
        plt.close()

        return fig

    #=====================================================================================================================================#

    def train(self):
        # Set data loader.
        data_loader = self.vcc_loader

        # define loss and output
        loss = {}
        output = {}
            
        # Start training.
        print('Start training...')
        start_time = time.time()
        epoch = 1
        start_iter = 0
        if self.continue_to_train:
            epoch = self.iter // 688
            start_iter = self.iter
        for i in range(start_iter, self.num_iters, 1):

            # =================================================================================== #
            #                             1. Preprocess input data                                #
            # =================================================================================== #
            data_iter = iter(data_loader)
            content, tgt_mel, nonpadding, spk_emb = next(data_iter)
            content = content.to(self.device)           # content:[B, T, H]
            tgt_mel = tgt_mel.to(self.device)           # mel:[B, H, T]
            nonpadding = nonpadding.to(self.device)     # nonpadding: [B, 1, T]
            spk_emb = spk_emb.to(self.device)           # spk_emb:[B, H]

            # =================================================================================== #
            #                               2. Train the generator                                #
            # =================================================================================== #
            self.G = self.G.train()

            if self.begin_post_flow == i:
                self.g_optimizer = self.g_optimizer2
                self.train_flag = True

            # import f-vae
            loss, output = self.G(tgt_mel.transpose(1, 2), cond=content, spk_emb=spk_emb,
                                  loss=loss, output=output, nonpadding=nonpadding, train_flag=self.train_flag)      # mel:[B, T, H], content:[B, T, H], nonpadding:[B, 1, T]

            loss['recon_vae'] = self.l1_loss(output['recon_vae'].transpose(1, 2), tgt_mel, nonpadding)

            # so far, loss includes: reconstruction L1 loss and kl loss
            if self.use_post_flow and self.train_flag:
                total_loss = loss['postflow']
            else:
                total_loss = loss['kl'] + loss['recon_vae']

            self.reset_grad()
            total_loss.backward()
            self.g_optimizer.step()

            # =================================================================================== #
            #                                 4. Miscellaneous                                    #
            # =================================================================================== #

            # Print out training information and add to tensorboard.
            if i == 0 or (i+1) % self.log_step == 0:
                # print out training information
                et = time.time() - start_time
                et = str(datetime.timedelta(seconds=et))[:-7]
                log = "Elapsed [{}], Epoch [{}], Iteration [{}/{}]".format(et, epoch, i+1, self.num_iters)
                # for tag in keys:
                for tag in loss:
                    log += ", {}: {:.4f}".format(tag, loss[tag])
                print(log)
            if (i+1) % 250 == 0:
                # add to tensorboard
                if not self.train_flag:
                    self.writer.add_scalar('training/kl loss', loss['kl'], i+1)
                self.writer.add_scalar('training/l1 loss', loss['recon_vae'], i+1)
                if self.use_post_flow and self.train_flag:
                    self.writer.add_scalar('training/postflow loss', loss['postflow'], i + 1)
                # print(self.g_optimizer)

            # add to validation audio to tensorboard
            if i == 0 or (i+1) % 1000 == 0:
                self.validating(i, self.train_flag)

            # save training model
            if i == 0 or (i+1) % 5000 == 0:
                model_name = 'model' + '{}'.format(i+1) + '.ckpt'
                torch.save(self.G.state_dict(), os.path.join(self.model_save, model_name))
                print('saving model at {} step'.format(i+1))

            if (i+1) % 688 == 0:
                epoch += 1

    def validating(self, i, train_flag):
        # load data
        data_iter = iter(self.validation_loader)
        val_content, val_mel, val_nonpadding, spk_emb = next(data_iter)

        val_content = val_content.to(self.device)
        val_mel = val_mel.to(self.device)
        val_nonpadding = val_nonpadding.to(self.device)
        spk_emb = spk_emb.to(self.device)

        # loss and output
        val_loss = {}
        val_output = {}

        # validation
        val_loss, val_output = self.G(tgt_mel=None, cond=val_content, spk_emb=spk_emb,
                                      loss=val_loss, output=val_output,
                                      nonpadding=val_nonpadding, infer=True, train_flag=train_flag)

        # compute l1 loss for 3 outputs
        if self.use_post_flow and train_flag:
            val_loss['recon_post_flow'] = self.l1_loss(val_output['recon_post_flow'].transpose(1, 2),
                                                       val_mel, val_nonpadding)
        val_loss['recon_vae'] = self.l1_loss(val_output['recon_vae'].transpose(1, 2),
                                             val_mel, val_nonpadding)

        # add to tensorboard
        self.writer.add_scalar('validation/recon_vae loss', val_loss['recon_vae'], i + 1)
        if self.use_post_flow and train_flag:
            self.writer.add_scalar('validation/recon_post_flow loss', val_loss['recon_post_flow'], i + 1)

        val_content, val_spk_emb, val_nonpadding, val_tag = self.val_show_dataloader()

        val_loss, val_output = self.G(None, cond=val_content, loss=val_loss, output=val_output, spk_emb=val_spk_emb,
                                      nonpadding=val_nonpadding, infer=True, train_flag=train_flag)

        output_vae = val_output['recon_vae'].transpose(1, 2)
        if self.use_post_flow and train_flag:
            output_post_flow = val_output['recon_post_flow'].transpose(1, 2)
            val_mel_input = torch.cat([output_vae, output_post_flow], dim=0)
        else:
            val_mel_input = output_vae

        # log
        for k in range(1, 9, 1):
            if k < 5:  # VC
                src_spk = 'gt_audio' + str(k)
                tgt_spk = 'gt_audio' + str(k + 1) if k < 4 else 'gt_audio' + str((k + 1) % 4)
            else:  # reconstruction
                src_spk = 'gt_audio' + str(k % 4) if k < 8 else 'gt_audio' + str(4)
                tgt_spk = src_spk
            # add mel-spectrogram to tensorboard
            self.writer.add_figure('val_mel/'+self.conf[src_spk][:4]+'->'+self.conf[tgt_spk][:4], self.plot_spectrogram(
                val_mel_input[k-1][:, :val_tag[k-1]].detach().cpu().numpy()), i+1)
            if self.use_post_flow and train_flag:
                self.writer.add_figure('val_mel/'+self.conf[src_spk][:4]+'->'+self.conf[tgt_spk][:4],
                                       self.plot_spectrogram(val_mel_input[k+7][:, :val_tag[k-1]].detach().cpu().numpy()), i+1)

            # add audio to tensorboard
            val_wav = self.validate.hifigan(val_mel_input[k-1][:, :val_tag[k-1]])
            self.writer.add_audio('val_audio/'+self.conf[src_spk][:4]+'->'+self.conf[tgt_spk][:4], val_wav[0],
                                  i+1, sample_rate=16000)
            if self.use_post_flow and train_flag:
                val_wav = self.validate.hifigan(val_mel_input[k+7][:, :val_tag[k-1]])
                self.writer.add_audio('val_audio/'+self.conf[src_spk][:4]+'->'+self.conf[tgt_spk][:4],
                                      val_wav[0], i+1, sample_rate=16000)

    def val_show_dataloader(self):
        content = []
        spk_emb = []
        tag = []
        for i in range(1, 9, 1):
            if i < 5:       # VC
                src_spk = 'gt_audio' + str(i)
                tgt_spk = 'gt_audio' + str(i+1) if i < 4 else 'gt_audio' + str((i+1) % 4)
            else:           # reconstruction
                src_spk = 'gt_audio' + str(i % 4) if i < 8 else 'gt_audio' + str(4)
                tgt_spk = src_spk
            content.append(torch.FloatTensor(pickle.load(open(os.path.join(
                self.conf['content_folder'], self.conf[src_spk]+'.pkl'), 'rb'))).squeeze(0).to(self.device))
            spk_emb.append(torch.FloatTensor(np.load(
                os.path.join(self.conf['spk_emb'], self.conf[tgt_spk][:-4]+'.npy'))).to(self.device))
            tag.append(content[i-1].shape[0])

        content_padding = pad_sequence(content, padding_value=0).permute(1, 0, 2)
        spk_emb_padding = pad_sequence(spk_emb, padding_value=0).T

        nonpadding = (content_padding.transpose(1, 2) != 0).float()[:, :].to(self.device)
        nonpadding = torch.mean(nonpadding, 1, keepdim=True)
        nonpadding[nonpadding > 0] = 1

        return content_padding, spk_emb_padding, nonpadding, tag
