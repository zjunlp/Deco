import math

import torch
import torch.nn.functional as F
from torch import nn
import numpy as np

def eval_accuracy(all_label_probs, test_labels, mode='diagonal_W', p_cf=None):
    # evaluate the accuracy with and without contextual calibration
    num_classes = all_label_probs.shape[1]
    if p_cf is None:
        # do not calibrate
        W = np.identity(num_classes)
        b = np.zeros([num_classes, 1])
    else:
        # calibrate
        if mode == "diagonal_W":
            W = np.linalg.inv(np.identity(num_classes) * p_cf)
            b = np.zeros([num_classes, 1])
        elif mode == "identity_W":
            W = np.identity(num_classes)
            b = -1 * np.expand_dims(p_cf, axis=-1)
        else:
            assert False

    correctness_list = []
    probs = []
    assert len(all_label_probs) == len(test_labels)
    for label_probs, true_label in zip(all_label_probs, test_labels):
        label_probs = label_probs / np.sum(label_probs) # normalize to 1

        calibrate_label_probs = np.matmul(W, np.expand_dims(label_probs, axis=-1)) + b
        calibrate_label_probs /= np.sum(calibrate_label_probs)
        probs.append(calibrate_label_probs)

        ans_label = np.argmax(calibrate_label_probs)
        if ans_label == true_label:
            correctness_list.append(1)
        else:
            correctness_list.append(0)
    return np.mean(correctness_list), probs

class ECELoss(nn.Module):
    """
    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).
    The input to this loss is the logits of a model, NOT the softmax scores.
    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:
    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |
    We then return a weighted average of the gaps, based on the number
    of samples in each bin
    The input to this loss is the logits of a model, NOT the softmax scores.
    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:
    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |
    We then return a weighted average of the gaps, based on the number
    of samples in each bin
    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """

    def __init__(self, n_bins=15):
        """
        n_bins (int): number of confidence interval bins
        """
        super(ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):  # softmaxes
        if type(logits) is list:
            logits = np.array(logits)
            labels = np.array(labels)
        if type(logits) is np.ndarray:
            logits = torch.from_numpy(logits).float().squeeze()
            labels = torch.from_numpy(np.array(labels))
            
        assert torch.is_tensor(logits) and torch.is_tensor(labels)
        
        softmaxes = F.softmax(logits, dim=-1)
        confidences, predictions = torch.max(softmaxes, -1)
        accuracies = predictions.eq(labels)
        
        ece = torch.zeros(1, device=labels.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        # print(torch.sum(accuracies).item() / logits.shape[0], ece)
        return ece
    
LABEL_DICT = {0: ['yes',], 1: ['no',]}
LABEL_TO_INT = {'yes': 0, 'no': 1}

def calibrate_label_dict(logits, tokenizer, label_dict=LABEL_DICT, top_k=10, apply_softmax=True, content_free_inputs=('N/A',)):
    probs = logits.float().cpu() if not apply_softmax else torch.softmax(logits, dim=-1).float().cpu()
    top_probs, top_tokens = torch.topk(probs, k=top_k)
    temp = {}
    for prob, token in zip(top_probs[0], top_tokens[0]):
        str_token = tokenizer.decode(token.item())
        str_token = str_token.lower().strip()
        if str_token not in temp.keys():
            temp[str_token] = prob.item()
        else:
            pass
    return temp

def get_prob_from_logits(top_token_probs, label_dict=LABEL_DICT):
    p_y = [0] * len(label_dict)
    for i, answers in label_dict.items():
        prob = 0
        for a in answers:
            a = a.lower()
            if a not in top_token_probs.keys():
                prob += 0
            else:
                prob += top_token_probs[a]
        p_y[i] = prob
    return p_y