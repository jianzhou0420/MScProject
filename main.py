# outer imports
import torch
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm
import os

# inner imports
from L0_Traning_Manager.trainner import Tranner
from utils import load_config





print('System is starting...')

parser = argparse.ArgumentParser(
    description='FullyExplicitNeRF'
)
parser.add_argument('--config', type=str, help='path to config file',default='./Support_Config/config_test.yaml')
parser.add_argument('--input_folder', type=str, help='path to input folder',default='./L1_Data_Manager/')
parser.add_argument('--output_folder', type=str, help='path to output folder',default='./Support_Output/')

args = parser.parse_args()

config = load_config(args.config)

output_ckpt_path=os.path.join(args.output_folder,'ckpt')
output_pictures_path=os.path.join(args.output_folder,'pictures')
test = Tranner(config)


test.train(200,1,1,output_ckpt_path)

# ckpt=''

# test.render_from_pt('./testall10.pt',output_pictures_path)
