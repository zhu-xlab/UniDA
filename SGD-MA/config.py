import yaml
import easydict
from os.path import join


class Dataset:
    def __init__(self, path_source, path_target, domains, files, prefix_source, prefix_target):
        self.path_source = path_source
        self.path_target = path_target
        self.domains = domains
        self.files = [join(path_source, files[0]), join(path_target, files[1])]
        self.prefix_source = prefix_source
        self.prefix_target = prefix_target



import argparse
parser = argparse.ArgumentParser(description='Code for *Universal Domain Adaptation*',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--config', type=str, default='Path/SGD-MA/RSSCN2NWPU-train-config.yaml', help='/config.yaml')

args = parser.parse_args()

config_file = args.config

args = yaml.load(open(config_file))

save_config = yaml.load(open(config_file))

args = easydict.EasyDict(args)

dataset = None
if args.data.dataset.name == 'RSSCN2NWPU':
    dataset = Dataset(
    path_source=args.data.dataset.source_root_path,
    path_target=args.data.dataset.target_root_path,
    domains=['RSSCN', 'NWPU'],
    files=[
        'RSSCN.txt',
        'NWPU-RESISC45.txt',
    ],
    prefix_source=args.data.dataset.source_root_path,
    prefix_target=args.data.dataset.target_root_path)
else:
    raise Exception('dataset {} not supported!'.format(args.data.dataset.name))

source_domain_name = dataset.domains[args.data.dataset.source]
target_domain_name = dataset.domains[args.data.dataset.target]
source_file = dataset.files[args.data.dataset.source]
target_file = dataset.files[args.data.dataset.target]
