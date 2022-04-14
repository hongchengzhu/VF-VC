import os
import pickle
import torch
import json
from hifigan.env import AttrDict
from hifigan.models import Generator as hifi_Generator


class validation(object):
    def __init__(self):
        """
        Inintialize hifigan
        """
        self.config = {}
        # hifigan
        self.hifigan_checkpoint = './checkpoint/hifigan/g_03280000'

        self.config_file = os.path.join(os.path.split(self.hifigan_checkpoint)[0], 'config.json')
        with open(self.config_file) as f:
            data = f.read()

        json_config = json.loads(data)
        self.h = AttrDict(json_config)

        torch.manual_seed(self.h.seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.h.seed)
            self.device = torch.device('cuda:0')
        else:
            self.device = torch.device('cpu')

        self.generator = hifi_Generator(self.h).to(self.device)

        self.state_dict_g = self.load_checkpoint(self.hifigan_checkpoint, self.device)
        self. generator.load_state_dict(self.state_dict_g['generator'])

        self.generator.eval()
        self.generator.remove_weight_norm()

    def load_checkpoint(self, filepath, device):
        assert os.path.isfile(filepath)
        print("Loading '{}'".format(filepath))
        checkpoint_dict = torch.load(filepath, map_location=device)
        print("Complete.")
        return checkpoint_dict

    def hifigan(self, x):
        with torch.no_grad():
            # x = torch.FloatTensor(x).to(self.device)
            # x = torch.unsqueeze(x, dim=-1)
            # x = x.permute(2, 1, 0)
            y_g_hat = self.generator(x)
            audio = y_g_hat.squeeze()
            audio = audio.view(-1).cpu().numpy().astype('float')

        return audio

    def forawrd(self, x):
        self.hifigan(x)


