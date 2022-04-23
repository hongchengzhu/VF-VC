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
        if config['is_training']:
            self.validate = validation()
            self.writer.add_figure('GT/LJ001-0001', self.plot_spectrogram(np.load('./feature/mel_hifigan/LJ001-0001.npy')))
            self.writer.add_audio('GT/LJ001-0001.wav', torch.tensor(sf.read('./wavs/LJ001-0001.wav')[0]).to(self.device),
                                  sample_rate=16000)

    def build_model(self, config):
        self.G = Generator(config)
        if self.continue_to_train is True:
            checkpoint = torch.load(config['load_model'], map_location={'cuda:1':'cuda:0'})
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
            epoch = self.iter // 204
            start_iter = self.iter
        for i in range(start_iter, self.num_iters, 1):

            # =================================================================================== #
            #                             1. Preprocess input data                                #
            # =================================================================================== #

            # Fetch data.
            # try:
            #     content, tgt_mel = next(data_iter)
            # except:
            data_iter = iter(data_loader)
            content, tgt_mel, nonpadding = next(data_iter)
            content = content.to(self.device)
            tgt_mel = tgt_mel.to(self.device)
            nonpadding = nonpadding.to(self.device)

            # =================================================================================== #
            #                               2. Train the generator                                #
            # =================================================================================== #

            self.G = self.G.train()

            if self.begin_post_flow == i:
                self.g_optimizer = self.g_optimizer2
                self.train_flag = True

            # import f-vae
            loss, output = self.G(tgt_mel.transpose(1, 2), cond=content,
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

            if (i+1) % 204 == 0:
                epoch += 1

    def validating(self, i, train_flag):
        # load data
        data_iter = iter(self.validation_loader)
        val_content, val_mel, val_nonpadding = next(data_iter)

        val_content = val_content.to(self.device)
        val_mel = val_mel.to(self.device)
        val_nonpadding = val_nonpadding.to(self.device)

        # loss and output
        val_loss = {}
        val_output = {}

        # validation
        val_loss, val_output = self.G(val_mel.transpose(1, 2), cond=val_content,
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

        # choose LJ001-0001.wav as show audio, [80, 604]
        chosen_mel = torch.FloatTensor(np.load('/home/hongcz/alab/feature/mel_hifigan_padding_alignment/LJ001-0001.npy')).unsqueeze(0).to(self.device)
        chosen_content = torch.FloatTensor(pickle.load(open('/home/hongcz/alab/feature/wav2vec2_padding_alignment/LJ001-0001.pkl', 'rb'))).to(self.device)
        chosen_nonpadding = (chosen_mel.transpose(1, 2) != 0).float()[:, :].to(self.device)
        chosen_nonpadding = torch.mean(chosen_nonpadding, 1, keepdim=True)
        chosen_nonpadding[chosen_nonpadding > 0] = 1

        val_loss, val_output = self.G(chosen_mel, cond=chosen_content, loss=val_loss, output=val_output,
                                      nonpadding=chosen_nonpadding, infer=True, train_flag=train_flag)

        output_vae = val_output['recon_vae'].transpose(1, 2)
        if self.use_post_flow and train_flag:
            output_post_flow = val_output['recon_post_flow'].transpose(1, 2)
            val_mel_input = torch.cat([output_vae, output_post_flow], dim=0)
        else:
            val_mel_input = output_vae

        #   # add mel-spectrogram to tensorboard
        self.writer.add_figure('RC_vae LJ001-0001.wav',
                               self.plot_spectrogram(val_mel_input[0][:, :-2].squeeze(0).detach().cpu().numpy()), i+1)
        if self.use_post_flow and train_flag:
            self.writer.add_figure('RC_post_flow LJ001-0001.wav',
                                   self.plot_spectrogram(val_mel_input[1][:, :-2].squeeze(0).detach().cpu().numpy()), i+1)

        #   # add audio to tensorboard
        val_wav = self.validate.hifigan(val_mel_input[:, :, :-2])
        self.writer.add_audio('RC_vae 001.wav', val_wav[0], i+1, sample_rate=16000)
        if self.use_post_flow and train_flag:
            self.writer.add_audio('RC_post_flow 001.wav', val_wav[1], i+1, sample_rate=16000)

