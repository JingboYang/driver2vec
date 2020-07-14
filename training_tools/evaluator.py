import copy
import re
import multiprocessing

import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix as skcm

from utils import *
from logger import Logger
from constants import *


INFO_RE = re.compile(r'Driver: (?P<driver>.*), Area: (?P<area>.*), index: (?P<index>.*)')

pool = None

def heavy_eval(is_heavy):
    def decorator(function):
        def wrapper(*args, **kwargs):
            # TODO Find a simple way to handle this business ...
            # If is eval, or if fast debug, or
            # is train and not heavy, or is train and heavy
            if ((args[0] != 'train') or args[1] or \
                (args[0] == 'train' and is_heavy and args[2]) or \
                (args[0] == 'train' and not is_heavy)):
                args = args[3:]
                result = function(*args, **kwargs)
                return result
            else:
                print(f'Skipping eval function {function.__name__}')
                return 'none', None, None
        return wrapper
    return decorator

def detach_tensor(tensor):
    if type(tensor) != np.ndarray:
        if type(tensor) == list:
            return np.ndarray(tensor)
        else:
            return tensor.cpu().detach().numpy()
    return tensor

def cash_to_tensor(tensor):
    if type(tensor) == np.ndarray:
        return torch.Tensor(tensor)
    return tensor

# Need wrapper for loss to handle different inputs
def loss_cross_entropy_wrapper(loss_inputs):
    pred = cash_to_tensor(loss_inputs['predictions'])
    target = cash_to_tensor(loss_inputs['ground_truth'])
    losses = F.cross_entropy(pred, target.long())

    return losses

# https://github.com/adambielski/siamese-triplet/blob/master/losses.py
def loss_triplet_wrapper(loss_inputs):
    # TODO Verify triplet loss
    margin = loss_inputs['triplet_margin']
    weight = loss_inputs['triplet_weight']
    pred = cash_to_tensor(loss_inputs['predictions'])
    target = cash_to_tensor(loss_inputs['ground_truth'])
    orig = cash_to_tensor(loss_inputs['other_info']['orig'])
    pos = cash_to_tensor(loss_inputs['other_info']['pos'])
    neg = cash_to_tensor(loss_inputs['other_info']['neg'])

    # This is to verify magnitude of embeddings
    loss_embedd = orig.norm(2) + pos.norm(2) + neg.norm(2)
    print(f'Triplet embedding magnitude loss {loss_embedd}')
    losses = F.triplet_margin_loss(orig, pos, neg, margin) * weight + \
                nn.CrossEntropyLoss()(pred, target.long()) * (1 - weight)

    return losses


# Evaluation metrics
def p_way_conf_matrix_mp_helper(arg):

    gt = arg['gt']
    orig_pred = copy.deepcopy(arg['pred'])
    mask_driver = arg['mask_driver']

    conf_pair = [0, 0]
    for j in range(mask_driver.shape[0]):
        pred = copy.deepcopy(orig_pred)       # (15, )
        mask_np = mask_driver[j].astype(int)

        pred[mask_np==np.array([0]).any()] = -1e5
        predicted = int(np.argmax(pred))

        if gt == predicted:
            conf_pair[0] += 1
        else:
            conf_pair[1] += 1
    return conf_pair

def p_way_conf_matrix_mp(predictions, ground_truth, mask_driver, p):
    """
    calculate pairwise accuracy based on a list of confusion matrices
    input: conf_matrices: list of numpy array, length is 15 choose 2
    returns accuracy
    helper function step 1
    generate a p by p confusion matrix - p is determined in dataset.py
    
    inputs:
        predictions: this is the scores generated from our model
                     dim()
        ground_truth: this is all actually driver i
        mask_driver: generated in dataset.py that enumerates all
                     possible NUM_DRIVERS-1 choose p-1
                     combinations to do p way conf matrix
    returns:
        conf_matrices: list of numpy array of size (p,p),
                       list length is NUM_DRIVERS-1 choose p-1
    """
    # 105 = 15 choose 2
    # Within a batch, iterate samples
    arguments = []
    for i in range(mask_driver.shape[0]):
        # Within a sample, iterate masks
        gt = int(ground_truth[i])   # scalar
        cur_arg = {'gt': gt,
                   'mask_driver': mask_driver[i],
                   'pred': predictions[i]}
        arguments.append(cur_arg)
    
    # TODO Probably should have a constant specifying # of CPU cores
    collected_matrices = pool.map(p_way_conf_matrix_mp_helper, arguments)
    
    conf_pair = [0, 0]
    for c in collected_matrices:
        conf_pair[0] += c[0]
        conf_pair[1] += c[1]
 
    if sum(conf_pair) == 0:
        print('conf_pair = 0, likely issue with area ID')
        import pdb; pdb.set_trace()

    acc = conf_pair[0] / (conf_pair[0] + conf_pair[1])
    return acc

def p_way_conf_matrix_single(predictions, ground_truth, mask_driver, p):
    """
    calculate pairwise accuracy based on a list of confusion matrices
    input: conf_matrices: list of numpy array, length is 15 choose 2
    returns accuracy
    helper function step 1
    generate a p by p confusion matrix - p is determined in dataset.py
    
    inputs:
        predictions: this is the scores generated from our model
                     dim()
        ground_truth: this is all actually driver i
        mask_driver: generated in dataset.py that enumerates all
                     possible NUM_DRIVERS-1 choose p-1
                     combinations to do p way conf matrix
    returns:
        conf_matrices: list of numpy array of size (p,p),
                       list length is NUM_DRIVERS-1 choose p-1
    """
    # TODO somehow track p here, p = 2
    # 105 = 15 choose 2
    conf_pair = [0, 0]
    # print(mask_driver.shape)
    # Within a batch, iterate samples
    for i in range(mask_driver.shape[0]):
        # Within a sample, iterate masks
        gt = int(ground_truth[i])   # scalar
        for j in range(mask_driver.shape[1]):
            pred = copy.deepcopy(predictions[i])       # (15, )
            mask_np = mask_driver[i][j].astype(int)

            pred[mask_np==np.array([0]).any()] = -1e5
            predicted = int(np.argmax(pred))

            if gt == predicted:
                conf_pair[0] += 1
            else:
                conf_pair[1] += 1

    acc = conf_pair[0] / (conf_pair[0] + conf_pair[1])
    return acc

def p_way_conf_matrix(predictions, ground_truth, mask_driver, p):
    """
    calculate pairwise accuracy based on a list of confusion matrices
    input: conf_matrices: list of numpy array, length is 15 choose 2
    returns accuracy
    helper function step 1
    generate a p by p confusion matrix - p is determined in dataset.py
    
    inputs:
        predictions: this is the scores generated from our model
                     dim()
        ground_truth: this is all actually driver i
        mask_driver: generated in dataset.py that enumerates all
                     possible NUM_DRIVERS-1 choose p-1
                     combinations to do p way conf matrix
    returns:
        conf_matrices: list of numpy array of size (p,p),
                       list length is NUM_DRIVERS-1 choose p-1
        
    
    """
    # TODO somehow track p here, p = 2
    # 105 = 15 choose 2
    conf_matrices = {}
    # print(mask_driver.shape)
    # Within a batch, iterate samples
    for i in range(mask_driver.shape[0]):
        # Within a sample, iterate masks
        for j in range(mask_driver.shape[1]):
            gt = int(ground_truth[i])   # scalar
            pred = copy.deepcopy(predictions[i])       # (15, )
            mask_np = mask_driver[i][j].astype(int)
            mask = tuple(mask_np)    # (15, )
            
            if mask not in conf_matrices:
                # we do not need the actual conf mat,
                # just need to track correct and incorrect predictions
                conf_matrices[mask] = np.zeros((2))

            pred[mask_np==np.array([0]).any()] = -1e5
            predicted = int(np.argmax(pred))

            if gt == predicted:
                conf_matrices[mask][0] += 1
            else:
                conf_matrices[mask][1] += 1

    return conf_matrices

def p_way_accuracy_helper(conf_matrices, p):
    """
    helper function step 2
    calculate p_way accuracy based on a list of confusion matrices 
    - p is determined in dataset.py
    input: 
        conf_matrices: list of numpy array of size (p,p),
        list length is NUM_DRIVERS-1 choose p-1
    returns 
        accuracy
    """
    # to save the sums
    # TODO Handle p-way (0, 1 is just some tuple)
    sum_mat = np.zeros((2))
    for cm in conf_matrices:
        sum_mat += conf_matrices[cm]

    # final acc based on sum_mat
    acc = sum_mat[0] / np.sum(sum_mat)
    return acc

def get_eval_masks(p, predictions, ground_truth):
    '''Helper function to extract constants from constants'''
    B = predictions.shape[0]
    mask_driver = list()
    
    MASK_DRIVERS = MaskGenerator.get_MaskGenerator().get_mask()

    for i in range(B):
        driver = int(ground_truth[i])
        combinations, _ = MASK_DRIVERS[driver][p].shape
        mask_driver_i = MASK_DRIVERS[driver][p].reshape(1,
                                                        combinations,
                                                        NUM_DRIVERS)
        # print("p={} driver {} mask_driver shape is {}".format(p, driver, mask_driver_i.shape))
        mask_driver.append(mask_driver_i)

    mask_driver = np.concatenate(mask_driver, axis=0)
    return mask_driver

@heavy_eval(False)
def p_way_accuracy(*args):
    """
    wrapper for average out accuracy based on all enumerations of
    masks of the same driver - p is passed through constants.py
    """
    p = args[0]
    info_for_eval = args[1]
    predictions = detach_tensor(info_for_eval['predictions'])
    ground_truth = detach_tensor(info_for_eval['ground_truth'])

    mask_driver = get_eval_masks(p, predictions, ground_truth)
    acc = p_way_conf_matrix_mp(predictions,
                               ground_truth,
                               mask_driver,
                               p)
    return 'scalar', f'{p}_way_accuracy', acc

@heavy_eval(False)
def area_accuracy(*args):
    """
    wrapper for average out accuracy based on all enumerations of
    masks of the same driver - p is passed through constants.py
    """
    p = args[0]   # area id
    info_for_eval = args[1]
    predictions = detach_tensor(info_for_eval['predictions'])
    ground_truth = detach_tensor(info_for_eval['ground_truth'])
    description_lst = detach_tensor(\
        info_for_eval['other_info']['data_info']['description'])
    
    indices = []
    for i, j in enumerate(description_lst):
        area_id = int(re.match(INFO_RE, j).groups()[1])
        if area_id == p:
            indices.append(i)

    area_pred = predictions[indices]
    area_gt = ground_truth[indices]
    area_od = np.random.randint(low=0, high=NUM_DRIVERS, size=len(area_gt))
    area_od = (area_od + (area_od == area_gt)) % NUM_DRIVERS

    gt_logits = area_pred[area_gt]
    od_logits = area_pred[area_od]

    acc = np.mean(gt_logits > od_logits)
    
    return 'scalar', f'area_{p}_accuracy', acc

@heavy_eval(False)
def one_hot_accuracy(*args):
    info_for_eval = args[0]
    predictions = detach_tensor(info_for_eval['predictions'])
    ground_truth = detach_tensor(info_for_eval['ground_truth'])
    predicted = np.argmax(predictions, axis=1)
    correct = np.sum(predicted == ground_truth)

    return 'scalar', 'one_hot_accuracy', correct / len(ground_truth)

@heavy_eval(False)
def triplet_accuracy(*args):
    info_for_eval = args[0]
    margin = info_for_eval['triplet_margin']
    ground_truth = detach_tensor(info_for_eval['ground_truth'])
    orig = detach_tensor(info_for_eval['other_info']['orig'])
    pos = detach_tensor(info_for_eval['other_info']['pos'])
    neg = detach_tensor(info_for_eval['other_info']['neg'])
    
    ap_distances = np.sqrt(np.sum(np.square(orig - pos), axis=1))
    an_distances = np.sqrt(np.sum(np.square(orig - neg), axis=1))

    ap_avg = np.mean(ap_distances)
    an_avg = np.mean(an_distances)
    print(f'Pos pair distance: {ap_avg}. Neg pair distance: {an_avg}')

    distance = ap_distances - an_distances + margin
    correct = np.sum(distance < 0)

    return 'scalar', 'triplet_accuracy', correct / len(ground_truth)

@heavy_eval(False)
def triplet_ratio(*args):
    info_for_eval = args[0]
    margin = info_for_eval['triplet_margin']
    ground_truth = detach_tensor(info_for_eval['ground_truth'])
    orig = detach_tensor(info_for_eval['other_info']['orig'])
    pos = detach_tensor(info_for_eval['other_info']['pos'])
    neg = detach_tensor(info_for_eval['other_info']['neg'])
    
    ap_distances = np.sqrt(np.sum(np.square(orig - pos), axis=1))
    an_distances = np.sqrt(np.sum(np.square(orig - neg), axis=1))

    ap_avg = np.mean(ap_distances)
    an_avg = np.mean(an_distances)

    return 'scalar', 'triplet_ratio', ap_avg / an_avg

@heavy_eval(False)
def triplet_diff_weight_ratio(*args):
    info_for_eval = args[0]
    margin = info_for_eval['triplet_margin']
    ground_truth = detach_tensor(info_for_eval['ground_truth'])
    orig = detach_tensor(info_for_eval['other_info']['orig'])
    pos = detach_tensor(info_for_eval['other_info']['pos'])
    neg = detach_tensor(info_for_eval['other_info']['neg'])
    
    ap_distances = np.sqrt(np.sum(np.square(orig - pos), axis=1))
    an_distances = np.sqrt(np.sum(np.square(orig - neg), axis=1))

    ap_avg = np.mean(ap_distances)
    an_avg = np.mean(an_distances)

    diff_weight_ratio = ((an_avg - ap_avg) /
                         np.mean(np.linalg.norm(orig, axis=1)))

    return 'scalar', 'triplet_diff_weight_ratio', diff_weight_ratio

@heavy_eval(False)
def per_driver_f1(*args):
    for_driver = args[0]
    info_for_eval = args[1]
    predictions = detach_tensor(info_for_eval['predictions'])
    ground_truth = detach_tensor(info_for_eval['ground_truth'])

    pr_driver_boolean = (ground_truth == for_driver)
    predicted = np.argmax(predictions, axis=1)
    pr_predicted = (predicted == for_driver)

    f1 = f1_score(pr_driver_boolean, pr_predicted, average='macro')

    return 'scalar', f'driver_{for_driver}_f1', f1

@heavy_eval(True)
def confusion_matrix(*args):
    info_for_eval = args[0]
    predictions = detach_tensor(info_for_eval['predictions'])
    ground_truth = detach_tensor(info_for_eval['ground_truth'])
    predicted = np.argmax(predictions, axis=1)
    cm = skcm(ground_truth, predicted)
    cm_plot = plot_confusion_matrix(cm)

    return 'image', f'confusion_matrix', cm_plot

@heavy_eval(True)
def tsne(*args):
    info_for_eval = args[0]
    ground_truth = detach_tensor(info_for_eval['ground_truth'])
    embeddings = detach_tensor(info_for_eval['other_info']['orig'])
    # TODO find a way to define max_length in a smart way
    max_length = 5000
    if len(embeddings) > max_length:
        Logger.get_unique_logger().log(f'Too many embeddings, '
                                       f'will randomly select {max_length}')
        indices = np.random.randint(0, len(embeddings), max_length)
        embeddings = embeddings[indices]
        # driver ground_truth
        ground_truth = ground_truth[indices]

    tsne_plot = plot_tsne(embeddings, ground_truth, perplexity=20)
    return 'image', f't-SNE', tsne_plot

@heavy_eval(True)
def tsne_collisions(*args):
    info_for_eval = args[0]
    ground_truth = detach_tensor(info_for_eval['ground_truth'])
    embeddings = detach_tensor(info_for_eval['other_info']['orig'])

    driver_to_collisions = np.array(INDEX_TO_COLLISION)
    # TODO find a way to define max_length in a smart way
    max_length = 5000
    if len(embeddings) > max_length:
        Logger.get_unique_logger().log(f'Too many embeddings, '
                                       f'will randomly select {max_length}')
        indices = np.random.randint(0, len(embeddings), max_length)
        embeddings = embeddings[indices]
        # driver ground_truth
        ground_truth = ground_truth[indices]
        
    # collisions based on ground truth  (driver)
    collisions_truth = driver_to_collisions[ground_truth]
    # print("embedding size is {}".format(embeddings.shape))
    # print("ground_truth size is {}".format(ground_truth.shape))

    tsne_plot = plot_tsne_collisions(embeddings, collisions_truth, perplexity=20)
    return 'image', f't-SNE-collisions', tsne_plot

pool = multiprocessing.Pool(4)
# TODO We might want multi-target stuff
class Evaluator():
    def __init__(self, args):

        self.device = args.misc_args.device
        self.heavy_log_steps = args.train_args.heavy_log_steps
        self.fast_debug = args.train_args.fast_debug
        # margin for triplet loss
        self.triplet_margin = args.train_args.triplet_margin
        self.define_loss(args.train_args.loss_fn_name)
        self.triplet_weight = args.train_args.triplet_weight

    # Loss related specifications
    def define_loss(self, loss_fn_name):
        if loss_fn_name == 'triplet':
            self.loss_fn = loss_triplet_wrapper
        elif loss_fn_name == 'cross_entropy':
            self.loss_fn = loss_cross_entropy_wrapper
        else:
            raise NotImplementedError(
                f'Loss function {loss_fn_name} not implemented.')

    def loss_backward(self, model, clipping_value):
        self.loss.backward()
        
        gradients = []
        weights = []
        for p in model.parameters():
            if p.grad is not None:
                g = p.grad.cpu().detach().numpy()
                pp = p.cpu().detach().numpy()
                gradients.append(g.flatten())
                weights.append(pp.flatten())
        gradient_norm = np.linalg.norm(np.concatenate(gradients, axis=0))
        weight_norm = np.linalg.norm(np.concatenate(weights, axis=0))

        # Some arbitrary limit but gradient should not be this big anyways
        # Basically, skip this step if gradient is crazy. This avoids
        # seeing NaN in model
        # TODO Maybe add NaN check?
        if (gradient_norm > 50 * clipping_value or
            np.isnan(gradient_norm)):
            large_gradient = True
        else:
            large_gradient = False

        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        return gradient_norm, weight_norm, large_gradient
    
    def evaluate(self, mode, optimizer, info_for_eval, eval_metrics):

        info_for_eval['triplet_margin'] = self.triplet_margin
        info_for_eval['triplet_weight'] = self.triplet_weight

        # Evaluations
        scalar_results = {}
        image_results = {}
        arg_0 = mode
        arg_1 = self.fast_debug
        arg_2 = (optimizer.total_step % self.heavy_log_steps) == 0
        for metric in eval_metrics:
            args = [arg_0, arg_1, arg_2]
            args.extend(metric[1])
            args.append(info_for_eval)

            result_type, eval_name, value = globals()[metric[0]](*args)
            
            if result_type == 'scalar':
                scalar_results[f'{mode}:{eval_name}'] = value
            elif result_type == 'image':
                image_results[f'{mode}:{eval_name}'] = value
            elif result_type == 'none':
                pass

        # Loss Function
        loss = self.loss_fn(info_for_eval)
        scalar_results[f'{mode}:loss'] = loss
        
        if mode == 'train':
            # Only keep loss when it is training
            self.loss = loss

            # Learning rate
            for pg in optimizer.optimizer.param_groups:
                learning_rate = pg['lr']
                scalar_results[f'{mode}:learning_rate'] = learning_rate

        return scalar_results, image_results
            
        

    
