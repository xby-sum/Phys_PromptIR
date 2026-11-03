import argparse
parser = argparse.ArgumentParser()
# Input Parameters
parser.add_argument('--cuda', type=int, default=0)
parser.add_argument('--epochs', type=int, default=120, help='maximum number of epochs to train the total model.')
parser.add_argument('--batch_size', type=int,default=8,help="Batch size to use per GPU")
parser.add_argument('--lr', type=float, default=2e-4, help='learning rate of encoder.')
parser.add_argument('--de_type', nargs='+', default=['denoise_15', 'denoise_25', 'denoise_50', 'derain', 'dehaze'],
                    help='which type of degradations is training and testing for.')
parser.add_argument('--patch_size', type=int, default=128, help='patchsize of input.')
parser.add_argument('--num_workers', type=int, default=16, help='number of workers.')
# path
parser.add_argument('--data_file_dir', type=str, default='data_dir/',  help='where clean images of denoising saves.')
parser.add_argument('--denoise_dir', type=str, default='data/Train/Denoise/',
                    help='where clean images of denoising saves.')
parser.add_argument('--derain_dir', type=str, default='data/Train/Derain/',
                    help='where training images of deraining saves.')
parser.add_argument('--dehaze_dir', type=str, default='data/Train/Dehaze/',
                    help='where training images of dehazing saves.')
parser.add_argument('--output_path', type=str, default="output/", help='output save path')
parser.add_argument('--ckpt_path', type=str, default="", help='checkpoint save path')
parser.add_argument("--wblogger",type=str,default="promptir",help = "Determine to log to wandb or not and the project name")
parser.add_argument("--ckpt_dir",type=str,default="train_ckpt",help = "Name of the Directory where the checkpoint is to be saved")
parser.add_argument("--num_gpus",type=int,default= 4,help = "Number of GPUs to use for training")
# Phase3: 物理约束控制
parser.add_argument("--use_physics_loss", action="store_true", help="是否使用物理损失")
parser.add_argument("--use_ppe", action="store_true", help="是否使用PPE模块")
parser.add_argument("--use_agf", action="store_true", help="是否使用AGF模块")
parser.add_argument("--freeze_ppe", action="store_true", help="冻结PPE模块")
parser.add_argument("--use_coral", action="store_true", help="启用PPE内部的CORAL对齐")
parser.add_argument("--use_ppe_perturb", action="store_true", help="启用PPE物理参数扰动")
parser.add_argument("--use_coral_ppe", action="store_true", help="启用PPE特征CORAL对齐")
parser.add_argument("--freeze_stage", type=int, default=1, choices=[0,1,2], help="0:正常训练,1:冻结主干,2:全微调")
parser.add_argument("--subset_size", type=int, default=None, help="训练子集大小")
options = parser.parse_args()
options = parser.parse_args()

