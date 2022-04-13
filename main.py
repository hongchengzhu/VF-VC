import os
import argparse
from solver_encoder import Solver
from data_loader import get_loader
from torch.backends import cudnn


def str2bool(v):
    return v.lower() in ('true')


def main(config):
    # For fast training.
    cudnn.benchmark = True

    # Data loader.
    vcc_loader = get_loader(config.batch_size)
    
    solver = Solver(vcc_loader, config)

    solver.train()
    # solver.validating()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # Training configuration.
    parser.add_argument('--batch_size', type=int, default=16, help='mini-batch size')
    parser.add_argument('--num_iters', type=int, default=100000, help='number of total iterations')
    
    # Miscellaneous.
    parser.add_argument('--task', type=str, default='w2m-non-parallel')
    parser.add_argument('--log_step', type=int, default=20)

    config = parser.parse_args()
    print(config)
    main(config)
