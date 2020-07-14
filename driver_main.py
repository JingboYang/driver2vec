import torch
from torch import nn

from arguments import ArgParser
from constants import *
from data import *
from logger import *
import models
from training_tools import Evaluator, Optimizer, Predictor
from training_tools import find_save_dir, save_progress, load_progress
from utils import *

# Define cloud storage here
cloudFS = GCStorage.get_CloudFS(PROJECT_ID, GC_BUCKET, CREDENTIAL_PATH)

# Parse command line arguments
parser = ArgParser()
args = parser.parse_args()

# Verify arguments
# check_args(args)

# Setup logger
logger = args.misc_args.logger

# Setup device
device = args.misc_args.device

# Select columns
selected_columns = sorted(select_columns(args.data_args.data_spec))
if args.model_args.wavelet:
    args.model_args.input_channels = len(selected_columns) * 2
else:
    args.model_args.input_channels = len(selected_columns)
log_string = (f'Num Columns={len(selected_columns)} -->> \n'
              + format_list(selected_columns,
                            fmt_func=lambda x: x.ljust(24)))
logger.log_text({'setup:columns': log_string}, 0)

# Setup dataset manager
ds_manager = DatasetManager(args, selected_columns)
data_loaders = ds_manager.get_dataloaders()

# Setup model
# model_init = models.__dict__[args.model_args.model_name]
if args.model_args.do_triplet:
    model_init = models.__dict__['TripletLoss']
else:
    model_init = models.__dict__[args.model_args.model_name]
model = model_init(**vars(args.model_args))
model = nn.DataParallel(model)   # Do not need DataParallel for only 1 device
model = model.to(device)

# Setup evaluator
evaluator = Evaluator(args)
eval_metrics = DATASET_EVAL_METRICS[args.data_args.dataset_version]

# Setup Predictor
predictor = Predictor(model, args.misc_args.device, args.train_args.fast_debug)

# Setup optimizer
optimizer = Optimizer(model.parameters(),
                      len(data_loaders['train']['train'].dataset),
                      args)

# Check model loading
save_path = find_save_dir(args)
if save_path is not None:
    logger.log(f'Load an old model from {args.misc_args.saved_model_name}')
    load_progress(save_path, model, optimizer, args)
else:
    logger.log('Train a new model')


# Define test as a function so we can run test more frequently
def do_test():
        
    if args.train_args.do_test:
        # The following should be the same as for normal evaluation
        cur_step = optimizer.total_step

        predictor.start_prediction(data_loaders['train']['train'])
        for loader_name in eval_metrics['test']:
            predictor_out = predictor.named_predict(
                loader_name,
                data_loaders['test'][loader_name],
                eval_metrics['test'][loader_name][1]
            )

            test_result = evaluator.evaluate(
                loader_name,
                optimizer,
                predictor_out,
                eval_metrics['test'][loader_name][0])
            scalar_results, image_results = test_result

            logger.log_scalars(scalar_results,
                                optimizer.total_step,
                                optimizer.epoch_step,
                                optimizer.cur_epoch)
            logger.log_images(image_results,
                                optimizer.total_step,
                                optimizer.epoch_step,
                                optimizer.cur_epoch)
    else:
        logger.log('Test skipped')


if args.train_args.do_train:
    model.train()

    while not optimizer.completed():

        for features, pos_features, neg_features, target, data_info\
            in data_loaders['train']['train']:

            # TODO Fix this by getting the right loss function rather than
            # skipping the ones with incorrect shape
            if len(features) == args.train_args.batch_size:
                features = features.to(device)
                target = target.to(device)
                with torch.set_grad_enabled(True):
                    predictions, other_info = model(features,
                                                    pos_features,
                                                    neg_features)
                    other_info['data_info'] = data_info
                    info_to_evaluate = {'predictions': predictions,
                                        'ground_truth': target,
                                        'other_info': other_info}

                    eval_result = evaluator.evaluate(
                                    'train', optimizer,
                                    info_to_evaluate,
                                    eval_metrics['train']['train'])
                    scalar_results, image_results = eval_result

                optimizer.zero_grad()
                # Compute gradient norm to prevent gradient explosion
                gradient_norm, weight_norm, large_gradient = \
                    evaluator.loss_backward(model,
                                            args.train_args.clipping_value)
                scalar_results[f'train:gradient_norm'] = gradient_norm
                scalar_results[f'train:weight_norm'] = weight_norm

                # TODO or not todo
                # Still useful if we ever see exploding gradient again
                # if large_gradient:
                #   logger.log_data(data_info['description'],
                #                   f'bad_samples_{optimizer.total_step}')

                logger.log_scalars(scalar_results,
                                optimizer.total_step,
                                optimizer.epoch_step,
                                optimizer.cur_epoch)
                logger.log_images(image_results,
                                optimizer.total_step,
                                optimizer.epoch_step,
                                optimizer.cur_epoch)

                # If gradient is large, just don't step that one
                if not large_gradient:
                    optimizer.step()
            else:
                logger.log(f'Skipping batch with size {len(features)} '
                           f'at total step {optimizer.total_step}')
            optimizer.end_iter()
            
            if (args.train_args.fast_debug or optimizer.total_step == 50 or
                (args.train_args.do_eval and
                 optimizer.total_step % args.train_args.eval_steps == 0)):

                cur_step = optimizer.total_step

                predictor.start_prediction(data_loaders['train']['train'])
                for loader_name in eval_metrics['eval']:
                    predictor_out = predictor.named_predict(
                        loader_name,
                        data_loaders['eval'][loader_name],
                        eval_metrics['eval'][loader_name][1]
                    )

                    eval_result = evaluator.evaluate(
                        loader_name,
                        optimizer,
                        predictor_out,
                        eval_metrics['eval'][loader_name][0])
                    scalar_results, image_results = eval_result

                    logger.log_scalars(scalar_results,
                                       optimizer.total_step,
                                       optimizer.epoch_step,
                                       optimizer.cur_epoch)
                    logger.log_images(image_results,
                                      optimizer.total_step,
                                      optimizer.epoch_step,
                                      optimizer.cur_epoch)
            
            if (args.train_args.fast_debug or
                (optimizer.total_step % args.train_args.save_steps == 0)):
                logger.log_text({'save:steps': f'Training progress saved at '
                                               f'step {optimizer.total_step}'},
                                optimizer.total_step)
                # save_progress(args.misc_args.model_save_dir,
                #               optimizer.total_step,
                #               model,
                #               optimizer)

                do_test()

        optimizer.end_epoch()
else:
    logger.log('Training skipped.')

