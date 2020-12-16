'''
@author Tian Shi
Please contact tshi@vt.edu
'''
import argparse

from LeafNATS.utils.utils import str2bool

parser = argparse.ArgumentParser()
'''
Use in the framework and cannot remove.
'''
parser.add_argument('--debug', type=str2bool, default=False, help='Debug?')
parser.add_argument('--task', default='train',
                    help='train | evaluate | visualization | keywords')

parser.add_argument(
    '--data_dir', default='../data/trip_binary', help='data dir')
parser.add_argument('--file_train', default='train.bert', help='train data.')
parser.add_argument('--file_val', default='dev.bert', help='dev data')
parser.add_argument('--file_test', default='test.bert', help='test data')

parser.add_argument('--n_epoch', type=int, default=10,
                    help='number of epochs.')
parser.add_argument('--batch_size', type=int, default=20, help='batch size.')
parser.add_argument('--checkpoint', type=int, default=500,
                    help='How often you want to save model?')

parser.add_argument('--continue_training', type=str2bool,
                    default=True, help='Do you want to continue?')
parser.add_argument('--train_base_model', type=str2bool, default=False,
                    help='True: Use Pretrained Param | False: Transfer Learning')
parser.add_argument('--is_lower', type=str2bool,
                    default=True, help='lower case for all words?')
'''
User specified parameters.
'''
parser.add_argument('--device', default="cuda:0", help='device')
# optimization
parser.add_argument('--learning_rate', type=float,
                    default=0.0005, help='learning rate.')
parser.add_argument('--grad_clip', type=float, default=2.0,
                    help='clip the gradient norm.')
# shared
parser.add_argument('--n_tasks', type=int, default=7, help='number of aspects')
parser.add_argument('--n_class', type=int, default=2, help='number of clsses')
parser.add_argument('--review_max_lens', type=int,
                    default=400, help='max length documents.')
# dropout
parser.add_argument('--drop_rate', type=float, default=0.2, help='dropout.')
# scheduler
parser.add_argument('--lr_schedule', type=str2bool,
                    default=True, help='Schedule learning rate.')
parser.add_argument('--step_size', type=int, default=2, help='---')
parser.add_argument('--step_decay', type=float, default=0.8, help='---')
# Visualization
parser.add_argument('--file_vis', default='test', help='vis data')
parser.add_argument('--base_model_dir',
                    default='../nats_results', help='base model dir')
parser.add_argument('--train_model_dir',
                    default='../nats_results', help='train model dir')
parser.add_argument('--best_model', type=int, default=1, help='---')

args = parser.parse_args()

'''
run model
'''
if args.task == 'evaluate':
    from LeafNATS.eval_scripts.eval_dmsc import evaluation
    evaluation(args)
if args.task == 'train':
    import torch
    args.device = torch.device(args.device)
    from .model import modelDMSC
    model = modelDMSC(args)
    model.train()
if args.task == 'visualization':
    import torch
    args.device = torch.device(args.device)
    from .model_view import modelView
    model = modelView(args)
    model.visualization()
if args.task == 'keywords':
    import torch
    args.device = torch.device(args.device)
    from .model_keywords import modelKeywords
    model = modelKeywords(args)
    model.keyword_extraction()
