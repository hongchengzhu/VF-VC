import yaml
from solver_encoder import Solver
from data_loader import get_loader, validation_get_loader
from torch.backends import cudnn


def main(config):
    # For fast training.
    cudnn.benchmark = True

    # Data loader.
    vcc_loader = get_loader(config)
    validation_loader = validation_get_loader(config)
    
    solver = Solver(vcc_loader, validation_loader, config)

    solver.train()
    # solver.validating()


if __name__ == '__main__':
    # config file is in './config/config.yaml'
    with open('./config/config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.load(f.read(), Loader=yaml.FullLoader)

    print(config)
    main(config)
