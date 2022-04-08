import os
import pickle
from model_vc import Generator
import torch
import numpy as np
import torch.nn.functional as F
import time
import librosa.display
import matplotlib.pyplot as plt
import datetime
import soundfile as sf
from torch.utils.tensorboard import SummaryWriter
from vc_validation import validation


class Solver(object):

    def __init__(self, vcc_loader, config):
        """Initialize configurations."""

        # Data loader.
        self.vcc_loader = vcc_loader

        # Model configurations.
        self.lambda_cd = config.lambda_cd
        self.dim_neck = config.dim_neck
        self.dim_emb = config.dim_emb
        self.dim_pre = config.dim_pre
        self.freq = config.freq

        # Training configurations.
        self.batch_size = config.batch_size
        self.num_iters = config.num_iters
        
        # Miscellaneous.
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device('cuda:0' if self.use_cuda else 'cpu')
        self.log_step = config.log_step

        # Build the model and tensorboard.
        self.build_model()
        self.writer = SummaryWriter('./log/tensorboard')
        self.model_save = './log/model'

        # validation
        self.validate = validation()
        self.val_mel = torch.tensor(np.load('./feature/mel/LJ001-0001.npy')).to(self.device).unsqueeze(0)
        self.val_content = pickle.load(open('./feature/wav2vec2/LJ001-0001.pkl', "rb")).unsqueeze(0)
        self.val_content_padding = torch.zeros([1, self.val_mel.shape[1], 768]).to(self.device)
        self.val_content_padding[:, :self.val_content.shape[1], :] = self.val_content
        # self.val_nonpadding = torch.zeros(self.val_mel.shape).to(self.device)
        # self.val_nonpadding = torch.mean(self.val_nonpadding.transpose(1, 2), 1, keepdim=True)
        self.val_nonpadding = torch.ones([1, 1, 128]).to(self.device)
        self.val_loss = {}
        self.val_output = {}
        self.writer.add_image('GT LJ001-0001', self.val_mel.transpose(1, 2)[:, :, 128:256])
        self.writer.add_audio('GT LJ001-0001.wav', torch.tensor(sf.read('./wavs/LJ001-0001.wav')[0]).to(self.device), sample_rate=16000)
            
    def build_model(self):

        # self.G = Generator(self.dim_neck, self.dim_emb, self.dim_pre, self.freq)
        self.G = Generator()
        self.g_optimizer = torch.optim.Adam(self.G.parameters(), 0.0001)
        self.G.to(self.device)

    def reset_grad(self):
        """Reset the gradient buffers."""
        self.g_optimizer.zero_grad()

    def mse_loss(self, input_a, input_b, nonpadding):
        return torch.square(input_a - input_b).sum() / sum(nonpadding[nonpadding == 1])

    def spec_show(self, x):
        # x = x.unsqueeze(0)
        x = x.cpu().numpy()
        plt.imshow(x)
        # plt.ylabel('Mel Frequency')
        # plt.xlabel('Time(s)')
        # plt.title('Mel Spectrogram')
        plt.show()

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
        for i in range(self.num_iters):

            # =================================================================================== #
            #                             1. Preprocess input data                                #
            # =================================================================================== #

            # Fetch data.
            try:
                content, tgt_mel = next(data_iter)
            except:
                data_iter = iter(data_loader)
                content, tgt_mel, nonpadding = next(data_iter)

            # =================================================================================== #
            #                               2. Train the generator                                #
            # =================================================================================== #
            
            self.G = self.G.train()

            # import f-vae
            self.spec_show(tgt_mel[0])
            loss, output = self.G(tgt_mel.transpose(1, 2), cond=content, loss=loss, output=output, nonpadding=nonpadding)
            loss['recon'] = self.mse_loss(output['x_recon'].transpose(1, 2), tgt_mel, nonpadding)
            # so far, loss includes: reconstruction L1 loss and kl loss
            total_loss = 3 * loss['kl'] + 10 * loss['recon']
            self.reset_grad()
            total_loss.backward()
            self.g_optimizer.step()

            # =================================================================================== #
            #                                 4. Miscellaneous                                    #
            # =================================================================================== #

            # Print out training information and add to tensorboard.
            if (i+1) % self.log_step == 0:
                # print out training information
                et = time.time() - start_time
                et = str(datetime.timedelta(seconds=et))[:-7]
                log = "Elapsed [{}], Iteration [{}/{}]".format(et, i+1, self.num_iters)
                # for tag in keys:
                for tag in loss:
                    log += ", {}: {:.4f}".format(tag, loss[tag])
                print(log)
                # add to tensorboard
                self.writer.add_scalar('kl loss', loss['kl'], i)
                self.writer.add_scalar('l1 loss', loss['recon'], i)

            # add to validation audio to tensorboard
            if i % 200 == 0:
                self.validating(i)


            # save training model
            if i % 4999 == 0:
                model_name = 'VF-VC_VAE_' + '{}'.format(i+1) + '.ckpt'
                torch.save(self.G.state_dict(), os.path.join(self.model_save, model_name))

    def validating(self, i):
        # x.transpose(1, 2)
        # model = Generator()
        # model.load_state_dict(torch.load('./log/model/VF-VC_VAE_5000.ckpt'))
        # model.to(self.device)
        # model.eval()

        self.val_loss, self.val_output = self.G(self.val_mel[:, 128:256, :], cond=self.val_mel[:, 128:256, :],
                                                loss=self.val_loss, output=self.val_output,
                                                nonpadding=self.val_nonpadding, infer=True)
        # self.val_output = output
        val_input = self.val_output['x_recon'].transpose(1, 2)
        vc_wav = self.validate.hifigan(val_input)

        name = './testwav/' + str(i) + '.wav'
        sf.write(name, vc_wav, samplerate=16000)

        # reconstruct
        self.writer.add_image('RC 001.wav', val_input, i+1)
        self.writer.add_audio('RC 001.wav',
                              vc_wav, i+1, sample_rate=16000)

