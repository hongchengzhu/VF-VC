import torch
import torch.nn as nn
from modules.VAE.fvae import FVAE
from modules.flow.glow_modules import Glow
import torch.distributions as dist
import torch.nn.functional as F


class ConvNorm(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=None, dilation=1, bias=True, w_init_gain='linear'):
        super(ConvNorm, self).__init__()
        if padding is None:
            assert(kernel_size % 2 == 1)
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = torch.nn.Conv1d(in_channels, out_channels,
                                    kernel_size=kernel_size, stride=stride,
                                    padding=padding, dilation=dilation,
                                    bias=bias)

        torch.nn.init.xavier_uniform_(
            self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, signal):
        conv_signal = self.conv(signal)
        return conv_signal


class Encoder(nn.Module):
    """Encoder module:
    """

    def __init__(self, dim_neck, dim_emb, freq):
        super(Encoder, self).__init__()
        self.dim_neck = dim_neck
        self.freq = freq

        convolutions = []
        for i in range(3):
            conv_layer = nn.Sequential(
                ConvNorm(80 + dim_emb if i == 0 else 512,
                         512,
                         kernel_size=5, stride=1,
                         padding=2,
                         dilation=1, w_init_gain='relu'),
                nn.BatchNorm1d(512))
            convolutions.append(conv_layer)
        self.convolutions = nn.ModuleList(convolutions)

        self.lstm = nn.LSTM(512, dim_neck, 2, batch_first=True, bidirectional=True)

    def forward(self, x, c_org):
        x = x.squeeze(1).transpose(2, 1)
        c_org = c_org.unsqueeze(-1).expand(-1, -1, x.size(-1))
        x = torch.cat((x, c_org), dim=1)

        for conv in self.convolutions:
            x = F.relu(conv(x))
        x = x.transpose(1, 2)

        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)
        out_forward = outputs[:, :, :self.dim_neck]
        out_backward = outputs[:, :, self.dim_neck:]

        codes = []
        for i in range(0, outputs.size(1), self.freq):
            codes.append(torch.cat((out_forward[:, i + self.freq - 1, :], out_backward[:, i, :]), dim=-1))

        return codes


class Generator(nn.Module):
    """Generator network."""
    def __init__(self, config):
        super(Generator, self).__init__()

        # transfer the content_size to hidden_size
        self.conv = torch.nn.Conv1d(config['hidden_in'], config['hidden_out'], kernel_size=1)

        # import f-vae
        self.fvae = FVAE(c_in_out=config['c_in_out'], hidden_size=config['hidden_size'], c_latent=config['c_latent'],
                         kernel_size=config['kernel_size'], enc_n_layers=config['enc_n_layers'],
                         dec_n_layers=config['dec_n_layers'], c_cond=config['c_cond'], strides=config['strides'],
                         use_prior_flow=config['use_prior_flow'], flow_hidden=config['flow_hidden'],
                         flow_kernel_size=config['flow_kernel_size'], flow_n_steps=config['flow_n_steps'],
                         flow_n_layers=config['flow_n_layers'], encoder_type=config['encoder_type'],
                         decoder_type=config['decoder_type'])

        # use Glow as post-net
        self.post_flow = Glow(
            config['c_in_out'], config['post_glow_hidden'], config['post_glow_kernel_size'], 1,
            config['post_glow_n_blocks'], config['post_glow_n_block_layers'],
            n_split=4, n_sqz=2,
            gin_channels=config['cond_hs'],
            share_cond_layers=config['post_share_cond_layers'],
            share_wn_layers=config['share_wn_layers'],
            sigmoid_scale=config['sigmoid_scale']
        )

        self.use_prior_flow = config['use_prior_flow']
        self.use_post_flow = config['use_post_flow']
        self.prior_dist = dist.Normal(0, 1)
        self.detach_postflow_input = config['detach_postflow_input']
        self.noise_scale = config['noise_scale']

    def forward(self, tgt_mel, cond, loss, output, nonpadding, spk_emb=None, train_flag=True, infer=False, noise_scale=1.0):
        """
        Content encoder & Speaker encoder (if use) as VF-VC Encoder, and the extracted representations are concated
        as the condition of modules (VF-VC Decoder) with VP-Flow enhanced. The flow-based model is used as Post-Net to
        enrich converted mel-spectrogram details.
        :param tgt_mel: [Batch, C_in_out, T]
        :param cond: [B, C_g, T]
        :param loss: empty dict return loss
        :param output: empty dict return output
        :param nonpadding: [B, C, T]
        :param spk_emb: speaker embedding in multi-speaker setting
        :param train_flag: set to True
        :param infer: train or infer
        :param noise_scale: 1.0
        :return:
        """

        # convert content to hidden before into vae
        T = cond.shape[1]
        spk_emb_padding = spk_emb.unsqueeze(1).repeat(1, T, 1)
        cond = torch.cat([cond, spk_emb_padding], -1)
        cond = self.conv(cond.transpose(1, 2))  # condition with 192-dim, [B, H, T]

        # f-vae encoder
        if not infer:
            z, kl, z_p, m_q, logs_q = self.fvae(tgt_mel, nonpadding, infer=False, cond=cond)
            output['z'] = z
            output['z_p'] = z_p
            output['m_q'] = m_q
            output['logs_q'] = logs_q
            if train_flag:
                kl = kl.detach()
            loss['kl'] = kl
        # f-vae decoder
        else:
            z = self.fvae(cond=cond, infer=True)
        x_recon = self.fvae.decoder(z, nonpadding=nonpadding, cond=cond).transpose(1, 2)
        output['recon_vae'] = x_recon * nonpadding.transpose(1, 2)  # vae with or w/o prior flow [B, T, H]
        if self.use_post_flow and train_flag:
            if not infer:
                loss['postflow'], output['z_postflow'] = self.run_post_flow(
                    tgt_mel, cond, nonpadding, infer, output, loss)
            else:
                output['recon_post_flow'] = self.run_post_flow(tgt_mel, cond, nonpadding, infer, output, loss)
        return loss, output

    def run_post_flow(self, tgt_mel, cond, nonpadding, infer, output, loss):
        x_recon = output['recon_vae'].transpose(1, 2)
        g = x_recon
        B, _, T = g.shape

        g = torch.cat([g, cond], 1)

        prior_dist = self.prior_dist
        if not infer:
            self.post_flow.train()
            # nonpadding = output['nonpadding'].transpose(1, 2)     # [B, 1, T]
            y_lengths = nonpadding.sum(-1)
            if self.detach_postflow_input:  # train only post_flow not bw to vae
                g = g.detach()
            tgt_mels = tgt_mel.transpose(1, 2)      # [B, H, T]
            z_postflow, ldj = self.post_flow(tgt_mels, nonpadding, g=g)
            ldj = ldj / y_lengths / 80
            postflow_loss = -prior_dist.log_prob(z_postflow).mean() - ldj.mean()
            if torch.isnan(postflow_loss):
                postflow_loss = None
            return postflow_loss, z_postflow
        else:
            nonpadding = torch.ones_like(x_recon[:, :1, :])
            z_post = torch.randn(x_recon.shape).to(g.device) * self.noise_scale
            x_recon, _ = self.post_flow(z_post, nonpadding, g, reverse=True)
            recon_post_flow = x_recon.transpose(1, 2)
            return recon_post_flow

