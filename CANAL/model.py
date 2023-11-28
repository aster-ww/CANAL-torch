# -*- coding: utf-8 -*-
import os
import gc
import json
import random
import math
import random
from functools import reduce
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.model_selection import train_test_split, ShuffleSplit, StratifiedShuffleSplit, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_recall_fscore_support, classification_report
import torch
from torch import nn
from torch.optim import Adam, SGD, AdamW
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torch.distributions import Beta
from sklearn.metrics import adjusted_rand_score
from performer_pytorch import PerformerLM
import scanpy as sc
import csv
import anndata as ad
from utils import *
from copy import deepcopy

def setup_seed(seed):
    """Choose a random seed to run the current model.

    :seed (int): random seed 
    """
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False

class SCDataset(Dataset):
    """
    Construct SCDataset from AnnData
    """
    def __init__(self, data, label, CLASS):
        super().__init__()
        self.data = data
        self.label = label
        self.CLASS = CLASS

    def __getitem__(self, index):
        rand_start = random.randint(0, self.data.shape[0] - 1)
        full_seq = self.data[rand_start].toarray()[0]
        full_seq[full_seq > (self.CLASS - 2)] = self.CLASS - 2  # 0 1 2 3 4 5
        full_seq = torch.from_numpy(full_seq).long()
        full_seq = torch.cat((full_seq, torch.tensor([0]))).cuda()  # .to(device)#
        seq_label = self.label[rand_start]
        return full_seq, seq_label

    def __len__(self):
        return self.data.shape[0]



def save_model(experiments, dataset, stage_num, is_final_stage, example_bank, label_dict,
               highly_variable_idx, model,  ckpt_dir):
    """Save current model, cell-type library as well as example bank for continual learning and evaluation.

    :experiments (str): name of experiments
    :stage_num (int): current stage of fine-tuning
    :is_final_stage (bool): whether current stage is the final stage or not;
    :example_bank (AnnData object): the selected examples for repeatedly reviewing
    :label_dict (array): cell type library with all the cell types that have appeared so far
    :highly_variable_idx (array): index of highly variable genes
    :model: the fine-tuned CANAL model after this stage
    :ckpt_dir: path to save the model
    """
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    if not is_final_stage:
        torch.save(
            {'model_state_dict': model.state_dict(),
             'annotation': label_dict,
             'example_bank': example_bank,
             'highly_variable_idx': highly_variable_idx},
            f'{ckpt_dir}{experiments}/stage_{stage_num}/{experiments}_finetune_dataset_{dataset}_CANAL.pth')
    else:
        torch.save(
            {'model_state_dict': model.state_dict(),
             'annotation': label_dict,
             'highly_variable_idx': highly_variable_idx},
            f'{ckpt_dir}{experiments}/stage_{stage_num}/{experiments}_finetune_dataset_{dataset}_CANAL.pth')
        


def example_bank_update(example_bank_previous, adata, new_model_annotation, embedding, prototype,
                        current_label_dict,current_label_set, current_label, each_class_num, current_stage):
    """Update current example bank via prototypes
    """
    example_bank_current = sc.AnnData()
    for i in range(len(current_label_set)):
        embedding_i = embedding[np.array(current_label) == current_label_set[i]]
        similarity_i = torch.matmul(embedding_i, prototype[i].t())
        k = min(each_class_num,embedding_i.size()[0])
        _, idx_near_i = torch.topk(similarity_i, dim=-1, largest=True, k=k)
        adata_i = adata[np.array(current_label) == current_label_set[i]][
            idx_near_i.detach().cpu().numpy()]#
        rank = np.arange(1, k + 1)
        adata_i.obs['celltype'] = current_label_dict[i]
        adata_i.obs['rank'] = rank
        adata_i.obs['stage'] = current_stage
        example_bank_current = adata_i if len(example_bank_current) == 0 else example_bank_current.concatenate(adata_i)
    if current_stage != 1:
        example_bank_final = sc.AnnData()
        example_bank_current = example_bank_previous.concatenate(example_bank_current)
        for i in range(len(new_model_annotation)):
            example_bank_current_i = example_bank_current[
                np.array(example_bank_current.obs['celltype']) == new_model_annotation[i]]
            delete_num_i = np.max([len(example_bank_current_i) - each_class_num, 0])
            # print(f"delete {delete_num_i} cells of cell type {new_model_annotation[i]}")
            if delete_num_i > 0:
                rank_max_i = sorted(np.array(example_bank_current_i.obs['rank']), reverse=True)[
                    delete_num_i - 1]
                if len(example_bank_final) == 0:
                    example_bank_final = example_bank_current_i[
                        np.array(example_bank_current_i.obs['rank']) < rank_max_i]
                else:
                    example_bank_final = example_bank_final.concatenate(
                        example_bank_current_i[
                            np.array(example_bank_current_i.obs['rank']) < rank_max_i])
            else:

                if len(example_bank_final) == 0:
                    example_bank_final = example_bank_current_i
                else:
                    example_bank_final = example_bank_final.concatenate(
                        example_bank_current_i)
    else:
        example_bank_final = example_bank_current
    print("\n")
    print("example bank after updating:")
    print(example_bank_final)
    print("\n")
    print("cell type composition of this example bank:")
    print(np.unique(example_bank_final.obs['celltype'], return_counts=True))
    print("\n")
    print("dataset composition from each stage of this example bank:")
    print(np.unique(example_bank_final.obs['stage'], return_counts=True))
    print("\n")
    return example_bank_final


class Identity(torch.nn.Module):
    """
    Construct network output layer
    """
    def __init__(self, SEQ_LEN, dropout=0., h_dim=64, out_dim=10):
        super(Identity, self).__init__()
        self.conv1 = nn.Conv2d(1, 1, (1, 200))
        self.act = nn.ReLU()
        self.fc1 = nn.Linear(in_features=SEQ_LEN, out_features=128, bias=True)
        self.act1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(in_features=128, out_features=h_dim, bias=True)
        self.act2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(in_features=h_dim, out_features=out_dim, bias=True)

    def forward(self, x):
        """
        Forward propagation
        """
        x = x[:, None, :, :]
        x = self.conv1(x)
        x = self.act(x)
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = self.act1(x)
        manifold = self.dropout1(x)
        x = self.fc2(manifold)
        x = self.act2(x)
        embedding = self.dropout2(x)
        x = self.fc3(embedding)
        return manifold, embedding, x

def get_embedding(data,CLASS,model):
    """get samples' representation of the penultimate layer
    """
    embedding = []
    for index in range(data.shape[0]):
        full_seq = data[index].toarray()[0]
        full_seq[full_seq > (CLASS - 2)] = CLASS - 2
        full_seq = torch.from_numpy(full_seq).long()
        full_seq = torch.cat((full_seq, torch.tensor([0]))).cuda()
        full_seq = full_seq.unsqueeze(0)
        _, embedding_i, _ = model(full_seq)
        embedding.append(embedding_i.cpu().numpy())
    return torch.Tensor(np.array(embedding)).squeeze(1).cuda()

class CANAL_model():
    """
    Construct the CANAL model, which continually adapts pre-trained language model to universal annotation of scRNA-seq data
    """

    def __init__(self, gpu_option = '0', CLASS = 7, EMBEDDING_DIM = 64, VALIDATE_EVERY = 1, PATIENCE = 10,
                 BATCH_SIZE = 14, EPOCHS = 100, LEARNING_RATE = 2e-4):
        super().__init__()
        self.gpu_option = gpu_option
        self.CLASS = CLASS
        self.EMBEDDING_DIM = EMBEDDING_DIM
        self.VALIDATE_EVERY = VALIDATE_EVERY
        self.PATIENCE = PATIENCE
        self.BATCH_SIZE = BATCH_SIZE
        self.EPOCHS = EPOCHS
        self.LEARNING_RATE = LEARNING_RATE


    def train(self, experiments, pre_dataset, dataset, adata, cell_type, current_stage, is_final_stage, ckpt_dir,
              rehearsal_size=1000, highly_variable_idx = None, lambda_KD = 0.1, SEED = 1):
        """Train the CANAL model at a certain stage

        :experiments (str): name of experiments
        :pre_dataset (str): name of previous dataset
        :dataset (str): name of current dataset
        :adata (AnnData object): current data used to fine-tune the model
        :cell_type (array): the corresponding cell types of current data
        :current_stage(int): which stage the model is fine-tuned
        :stage_num (int): current stage of fine-tuning
        :is_final_stage (bool): whether current stage is the final stage or not
        :ckpt_dir: path to save the model
        :rehearsal_size (int): the number of cells in total we can preserve in the example bank, default = 1000
        :highly_variable_idx (array): index of highly variable genes, should be provided if this is the initial stage
        :lambda_KD (float): trainin weight for the representation knowledge distillation loss, default = 0.1
        :gpu_option (int): which device to run the model, default = '0'
        :SEED (int): which random seed is chosen to run the current model, default = 1
        """
        setup_seed(SEED)
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = self.gpu_option
        current_label_dict, label = np.unique(np.array(cell_type), return_inverse=True)
        print("current data:", adata)
        print("\n")
        print(np.unique(np.array(cell_type), return_counts=True))
        current_label = label
        label = torch.from_numpy(label)
        current_label_set = np.unique(current_label)
        data = adata.X
        data_copy = data
        print("model constructing begin!\n")
        lambda_representation_dl = 0.0
        if current_stage == 1:
            new_model_annotation = current_label_dict
            model_path = f'./{ckpt_dir}/panglao_pretrain.pth'
            ckpt = torch.load(model_path)
            SEQ_LEN = len(highly_variable_idx) + 1
            model = PerformerLM(num_tokens=self.CLASS, dim=200, depth=6, max_seq_len=SEQ_LEN,
                                highly_gene_idx=highly_variable_idx, heads=10, local_attn_heads=0,
                                g2v_position_emb=True)
            model_dict = model.state_dict()
            example_bank_previous = sc.AnnData()
            for i in ckpt['model_state_dict']:
                if i == "pos_emb.emb.weight":
                    chosen_idx = np.append(highly_variable_idx, [16906])
                    model_dict[i] = ckpt['model_state_dict'][i][chosen_idx,]
                else:
                    model_dict[i] = ckpt['model_state_dict'][i]
            model.load_state_dict(model_dict)

            for param in model.parameters():
                param.requires_grad = False
            for param in model.norm.parameters():
                param.requires_grad = True
            for param in model.performer.net.layers[-2].parameters():
                param.requires_grad = True
            model.to_out = Identity(SEQ_LEN=SEQ_LEN, dropout=0., h_dim=self.EMBEDDING_DIM, out_dim=current_label_dict.shape[0])
            for param in model.to_out.parameters():
                param.requires_grad = True

        else:
            model_path = f'{ckpt_dir}{experiments}/stage_{current_stage-1}/{experiments}_finetune_dataset_{pre_dataset}_CANAL.pth'
            ckpt = torch.load(model_path)
            highly_variable_idx = ckpt['highly_variable_idx']
            SEQ_LEN = len(highly_variable_idx) + 1
            old_model_annotation = ckpt['annotation']
            # print("previous mapping function betweem model outputs and cell types:", old_model_annotation)
            new_add_cell_type = [i for i in current_label_dict if i not in old_model_annotation]
            print("new cell types:", new_add_cell_type)
            new_model_annotation = np.concatenate([old_model_annotation, new_add_cell_type], axis=0)
            label = np.array([list(new_model_annotation).index(cell_type[i]) for i in range(len(label))])
            # print("new mapping function betweem model outputs and cell types:", new_model_annotation)
            lambda_representation_dl = lambda_KD * len(old_model_annotation) / len(new_model_annotation)
            model = PerformerLM(num_tokens=self.CLASS, dim=200, depth=6, max_seq_len=SEQ_LEN,
                                highly_gene_idx=highly_variable_idx, heads=10, local_attn_heads=0,
                                g2v_position_emb=True)
            model.to_out = Identity(SEQ_LEN=SEQ_LEN, dropout=0., h_dim=self.EMBEDDING_DIM,
                                    out_dim=len(new_model_annotation))
            model_dict = model.state_dict()
            example_bank_previous = ckpt['example_bank']
            print("example bank for experience replay:", example_bank_previous)
            for i in ckpt['model_state_dict']:
                if "to_out.fc3" in i:
                    previous_out_dim = ckpt['model_state_dict'][i].shape[0]
                    model_dict[i][:previous_out_dim] = ckpt['model_state_dict'][i]
                else:
                    model_dict[i] = ckpt['model_state_dict'][i]

            model.load_state_dict(model_dict)

            for param in model.parameters():
                param.requires_grad = False
            for param in model.norm.parameters():
                param.requires_grad = True
            for param in model.performer.net.layers[-2].parameters():
                param.requires_grad = True
            for param in model.to_out.parameters():
                param.requires_grad = True

        model_old = deepcopy(model)
        model_old = model_old.cuda()
        for p in model_old.parameters():
            p.requires_grad = False
        print("model constructing finished!\n")
        if current_stage != 1:
            data = sparse.vstack([data, example_bank_previous.X])
            example_bank_previous_label = [list(new_model_annotation).index(i) for i in np.array(example_bank_previous.obs['celltype'])]
            label = np.concatenate([label, np.array(example_bank_previous_label)], 0)
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
        for index_train, index_val in sss.split(data, label):
            data_train, label_train = data[index_train], label[index_train]
            data_val, label_val = data[index_val], label[index_val]
            print("label train:", np.unique(label_train), len(np.unique(label_train)))
            print("label val:", np.unique(label_val), len(np.unique(label_val)))
            train_dataset = SCDataset(data_train, label_train, self.CLASS)
            val_dataset = SCDataset(data_val, label_val, self.CLASS)

        train_loader = DataLoader(train_dataset, batch_size=self.BATCH_SIZE, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=self.BATCH_SIZE, shuffle=True, drop_last=True)
        model = model.cuda()
        # optimizer
        optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=self.LEARNING_RATE, amsgrad=True)  #
        scheduler = CosineAnnealingWarmupRestarts(
            optimizer,
            first_cycle_steps=15,
            cycle_mult=2,
            max_lr=self.LEARNING_RATE,
            min_lr=1e-6,
            warmup_steps=5,
            gamma=0.9
        )
        loss_fn = CrossEntropyLabelSmooth(len(new_model_annotation),epsilon=0.1).cuda()
        loss_mse = torch.nn.MSELoss(reduce=True)
        trigger_times = 0
        max_acc = 0.0
        if current_stage == 1:
            print(
                f'  ==  Begin finetuning: | Initial stage | Current stage: {current_stage} | CANAL | Dataset: {experiments} {dataset} ==')
        elif not is_final_stage:
            print(
                f'  ==  Begin finetuning: | Incrmental stage | Current stage: {current_stage} | CANAL | Dataset: {experiments} {dataset}  ==')
        else:
            print(
                f'  ==  Begin finetuning: | Final stage | Current stage: {current_stage} | CANAL | Dataset: {experiments} {dataset} ==')

        for i in range(1, self.EPOCHS + 1):
            model.train()
            running_loss_cls = 0.0
            running_loss_representation_dl = 0.0
            cum_acc = 0.0
            for index, (data, labels) in enumerate(train_loader):
                index += 1
                data, labels = data.cuda(), labels.cuda()
                _, representation, logits = model(data)
                with torch.no_grad():
                    _, representation_old, _ = model_old(data)
                loss_cls = loss_fn(logits, labels)
                loss_representation_dl = torch.tensor(0.0).cuda()
                if current_stage != 1:
                    loss_representation_dl = loss_mse(representation,representation_old)
                    loss = lambda_representation_dl * loss_representation_dl + loss_cls
                else:
                    loss = loss_cls
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), int(1e10))
                optimizer.step()
                optimizer.zero_grad()
                running_loss_cls += loss_cls.item()
                running_loss_representation_dl += loss_representation_dl.item()
                softmax = nn.Softmax(dim=-1)
                final = softmax(logits)
                final = final.argmax(dim=-1)
                pred_num = labels.size(0)
                correct_num = torch.eq(final, labels).sum(dim=-1)
                cum_acc += torch.true_divide(correct_num, pred_num).mean().item()
            epoch_loss_cls = running_loss_cls / index
            epoch_loss_representation_dl = running_loss_representation_dl / index
            epoch_acc = 100 * cum_acc / index
            print(
                f'    ==  Epoch: {i} | Classification Loss: {epoch_loss_cls:.6f} | Representation DL Loss: {epoch_loss_representation_dl:.6f}  | Accuracy: {epoch_acc:6.4f}%  ==')
            scheduler.step()

            if i % self.VALIDATE_EVERY == 0:
                model.eval()
                running_loss_cls = 0.0
                running_loss_representation_dl = 0.0
                predictions = []
                truths = []
                with torch.no_grad():
                    for index, (data_v, labels_v) in enumerate(val_loader):
                        index += 1
                        data_v, labels_v = data_v.cuda(), labels_v.cuda()
                        _, representation, logits = model(data_v)
                        with torch.no_grad():
                            _, representation_old, _ = model_old(data_v)
                        loss_cls = loss_fn(logits, labels)
                        loss_representation_dl = torch.tensor(0.0).cuda()
                        if current_stage!=1:
                            loss_representation_dl = loss_mse(representation, representation_old)
                        running_loss_cls += loss_cls.item()
                        running_loss_representation_dl += loss_representation_dl.item()
                        softmax = nn.Softmax(dim=-1)
                        final_prob = softmax(logits)
                        final = final_prob.argmax(dim=-1)
                        predictions += list(final.cpu().numpy())
                        truths += list(labels_v.cpu().numpy())

                    del data_v, labels_v, logits, final_prob, final
                    no_drop = np.array(predictions) != -1
                    predictions = np.array(predictions)[no_drop]
                    truths = np.array(truths)[no_drop]
                    cur_acc = accuracy_score(truths, predictions)
                    f1 = f1_score(truths, predictions, average='macro')
                    val_loss_cls = running_loss_cls / index
                    val_loss_representation_dl = running_loss_representation_dl / index
                    print(
                        f'    ==  Epoch: {i} | Validation CLS Loss: {val_loss_cls:.6f} | Validation Representation DL Loss: {val_loss_representation_dl:.6f} | F1 Score: {f1:.6f}  == | Current accuracy: {cur_acc:.6f}  ==')
                    print(confusion_matrix(truths, predictions))
                    if cur_acc > max_acc:
                        max_acc = cur_acc
                        trigger_times = 0
                        print("Patience:", trigger_times, "/", self.PATIENCE)
                        best_model = deepcopy(model)
                    else:
                        trigger_times += 1
                        print("Patience:", trigger_times, "/", self.PATIENCE)
                        if trigger_times > self.PATIENCE:
                            example_bank_final = sc.AnnData()
                            if not is_final_stage:
                                each_class_num = int(rehearsal_size / (len(new_model_annotation)))
                                prototype = best_model.state_dict()['to_out.fc3.weight']
                                embedding = get_embedding(data_copy, self.CLASS, best_model)
                                embedding_ = embedding.detach().clone()
                                prototype_ = prototype.detach().clone()
                                example_bank_final = example_bank_update(example_bank_previous, adata, new_model_annotation, embedding_, prototype_,
                                                                         current_label_dict,current_label_set, current_label, each_class_num, current_stage)
                            save_model(experiments, dataset, current_stage, is_final_stage, example_bank_final,
                                       new_model_annotation, highly_variable_idx, best_model, ckpt_dir)
                            break
            del predictions, truths

    def predict(self, adata_predict, ckpt_dir, experiments, stage_num, dataset, novel=False, temperature=0.8):
        """
        Predict cell types of an unlabelled dataset using the CANAL model, which has completed fine-tuning after certain stage

        :adata_predict (AnnData object): unlabeled test data we need to annotate
        :ckpt_dir: path of the saved fine-tuned CANAL model
        :experiments (str): name of experiments
        :stage_num (int): which stage of fine-tuning the CANAL model has completed
        :dataset (str): name of the dataset used during the given stage
        :novel (bool): whether to detect novel cells in the test data, default value is False. If True, we will annotate cells as "Unassigned", if their uncertainty score is larger than the automatic determined threshold.
        :temperature (float): the temperature parameter in the energy function if novel cell detection is needed. The default value is 0.8
        
        :return: an array, which provides predicted cell types of the test dataset

        """
        data = adata_predict.X
        model_path = f'{ckpt_dir}{experiments}/stage_{stage_num}/{experiments}_finetune_dataset_{dataset}_CANAL.pth'
        # print("Model for usage:", model_path)
        print(
            f'    ==  Begin predicting after {stage_num} fine-tuning stages: | Experiments: {experiments} ==')
        ckpt = torch.load(model_path)
        highly_variable_idx = ckpt['highly_variable_idx']
        SEQ_LEN = len(highly_variable_idx) + 1
        annotation = ckpt['annotation']
        print("Annotation:", annotation, '\n')
        model = PerformerLM(
            num_tokens=self.CLASS,
            dim=200,
            depth=6,
            max_seq_len=SEQ_LEN,
            highly_gene_idx=highly_variable_idx,
            heads=10,
            local_attn_heads=0,
            g2v_position_emb=True
        )
        out_dim = ckpt['model_state_dict']["to_out.fc3.bias"].size()[0]
        model.to_out = Identity(SEQ_LEN=SEQ_LEN, dropout=0., h_dim=64, out_dim=out_dim)
        model.load_state_dict(ckpt['model_state_dict'])
        for param in model.parameters():
            param.requires_grad = False
        model = model.cuda()
        batch_size = data.shape[0]
        model.eval()
        pred_finals = []
        m = Beta(torch.tensor([1.0]), torch.tensor([1.0]))
        if novel==True:
            confidence_score = []
            energy_score = []
            threshold_energy = []
            threshold_confidence = []
        with torch.no_grad():
            for index in range(batch_size):
                full_seq = data[index].toarray()[0]
                full_seq[full_seq > (self.CLASS - 2)] = self.CLASS - 2
                full_seq = torch.from_numpy(full_seq).long()
                full_seq = torch.cat((full_seq, torch.tensor([0]))).cuda()
                full_seq = full_seq.unsqueeze(0)
                manifold1,_, pred_logits = model(full_seq)
                softmax = nn.Softmax(dim=-1)
                pred_prob = softmax(pred_logits)
                pred_final = pred_prob.argmax(dim=-1)
                if novel==True:
                    confidence = np.amax(np.array(pred_prob.cpu()), axis=-1)
                    confidence_score.append(confidence)
                    energy = torch.exp(pred_logits / temperature).sum(-1)
                    energy = -torch.log(energy) * temperature
                    energy_score.append(energy.cpu().numpy())
                    index_rand = random.randint(0, batch_size - 1)
                    full_seq2 = data[index_rand].toarray()[0]
                    full_seq2[full_seq2 > (self.CLASS - 2)] = self.CLASS - 2
                    full_seq2 = torch.from_numpy(full_seq2).long()
                    full_seq2 = torch.cat((full_seq2, torch.tensor([0]))).cuda()
                    full_seq2 = full_seq2.unsqueeze(0)
                    manifold2, _, _ = model(full_seq2)
                    weight = (m.sample([manifold1.size()[0], manifold1.size()[1]])).cuda()
                    pred_logits_mix = model.to_out.fc2((weight * manifold1 + (1 - weight) * manifold2))
                    pred_logits_mix = model.to_out.act2(pred_logits_mix)
                    pred_logits_mix = model.to_out.dropout2(pred_logits_mix)
                    pred_logits_mix = model.to_out.fc3(pred_logits_mix)
                    pred_prob_mix = softmax(pred_logits_mix)
                    confidence_mix = np.amax(np.array(pred_prob_mix.cpu()), axis=-1)
                    energy_mix = torch.exp(pred_logits_mix / temperature).sum(-1)
                    energy_mix = -torch.log(energy_mix) * temperature
                    threshold_energy.append(energy_mix.cpu().numpy())
                    threshold_confidence.append(confidence_mix)
                pred_finals.append(int(pred_final.cpu().numpy()))
        pred_list = annotation[pred_finals].tolist()
        if novel==True:
            threshold_energy = 1-((threshold_energy-np.min(threshold_energy))/(np.max(threshold_energy)-np.min(threshold_energy)))
            threshold = (np.mean(threshold_energy)+np.mean(threshold_confidence))/2
            energy_score = ((energy_score-np.min(energy_score))/(np.max(energy_score)-np.min(energy_score)))
            total_score=[]
            for index in range(batch_size):
                total_score_i = (energy_score[index]+1-confidence_score[index])/2
                total_score.append(total_score_i)
                if total_score_i > threshold:
                   pred_list[index] = 'Unassigned'
        pred_cell_type = np.array(pred_list)
        return pred_cell_type

    def evaluation(self, pred_cell_type, true_celltype, novel_celltype=None):
        """
        If the test dataset has ground-truth cell-type labels, evaluate the performance of the CANAL model.
        If no novel cells in the test data, we will exhibit total annotation accuracy, F1 score as well as ARI;
        if novel cells exist in the test data, we will exhibit H-score, total annotation accuracy, annotation accuracy of known cells as well as accuracy of unknown cells

        :pred_cell_type (array): predicted cell types of the test dataset via the CANAL model
        :true_cell_type (array): true cell types of the test dataset
        :novel_celltype (array): novel cell type of the test dataset that have never appeared in the fine-tuning data stream. The default value is None.
        """
        print("Begin evaluation!\n")
        if novel_celltype != None:
            true_celltype = [i if i not in novel_celltype else 'Unassigned' for i in true_celltype]
            known_idx = []
            unknown_idx = []
            for index in range(len(true_celltype)):
                if true_celltype[index] == 'Unassigned':
                    unknown_idx.append(index)
                else:
                    known_idx.append(index)
            total_accuracy = np.round(np.mean(pred_cell_type == np.array(true_celltype)), 4)
            known_accuracy = np.mean(pred_cell_type[known_idx] == np.array(true_celltype)[known_idx])
            unknown_accuracy = np.mean(pred_cell_type[unknown_idx] == np.array(true_celltype)[unknown_idx])
            h_score = 2 * known_accuracy * unknown_accuracy / (known_accuracy + unknown_accuracy)
            print(
                f'  ==  H-score: {h_score:.6f} ==|== Predict total accuracy: {total_accuracy:.6f}  ==\n')
            print(
                f'  ==  Predict known accuracy: {known_accuracy:.6f}  ==|== Predict unknown accuracy: {unknown_accuracy:.6f}  ==',
                '\n')
        else:
            total_accuracy = np.round(np.mean(pred_cell_type == np.array(true_celltype)), 4)
            f1_marco = np.round(f1_score(np.array(true_celltype), pred_cell_type, average='macro'), 4)
            ARI = np.round(adjusted_rand_score(pred_cell_type, np.array(true_celltype)), 4)
            print(f'  ==  Predict total accuracy: {total_accuracy:.6f} ==|== F1 Score: {f1_marco:.6f}  ==|==  ARI: {ARI:.6f} ==',
                  '\n')
            print("Confusion matrix:")
            print(confusion_matrix(true_celltype, pred_cell_type))


