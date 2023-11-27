.. inclusion-marker-do-not-remove
CANAL: Continually adapting pre-trained language model to universal annotation of single-cell RNA-seq data
==========================================================================================================

PyTorch implementation of CANAL, a universal cell-type annotation tool
that continuously fine-tunes a pretrained language model trained on a
large amount of unlabeled scRNA-seq data, as new well-labeled data
emerges.

.. raw:: html

   <p align="center">

.. raw:: html

   </p>

File Descriptions and data requirement
======================================

Cell-type annotation of single-cell RNA-seq (scRNA-seq) data is a
hallmark of biomedical research and clinical application. Current
annotation tools usually assume the simultaneous acquisition of
well-annotated data, but without the ability to expand knowledge from
new data. Yet, such tools are inconsistent with the continuous emergence
of scRNA-seq data, calling for a continuous cell-type annotation model.
In addition, by their powerful ability of information integration and
model interpretability, transformer-based pre-trained language models
have led to breakthroughs in single-cell biology research. We herein
propose a universal cell-type annotation tool, called CANAL, that
continuously fine-tunes a pretrained language model trained on a large
amount of unlabeled scRNA-seq data, as new well-labeled data emerges.
For model inputs, we designed an experience replay schema that
repeatedly reviews previous vital examples in current training stages
via a dynamic example bank with fixed buffer size. This example bank is
class-balanced and good at memorizing and consolidating
cell-type-specific information, as well as patterns of rare cell types.
For model outputs, we utilize representation knowledge distillation to
regularize the divergence between previous and current models, resulting
in the preservation of knowledge learned from past training stages.
Moreover, our universal annotation framework considers new cell types
during both the fine-tuning and testing stages.

# Hyper-parameters and recommended settings

   -  lambda: default 0.1, the strength of representation distillation
      loss

   -  L: default 1000, the size of example bank

Data Availability
=================

+------------------+---------------------------------------------------+
| Link             | Description                                       |
+==================+===================================================+
| http             | Datasets of the pancreas experiemnts              |
| s://drive.google |                                                   |
| .com/drive/folde |                                                   |
| rs/1BMf-N-k-3aCE |                                                   |
| Y7CJvUcK9nZZ2UD7 |                                                   |
| p3C0?usp=sharing |                                                   |
+------------------+---------------------------------------------------+
| http             | Datasets of the cross-tissue experiemnts          |
| s://drive.google |                                                   |
| .com/drive/folde |                                                   |
| rs/1CaBySV_EFAPP |                                                   |
| rlpSevEewFds5cjJ |                                                   |
| xC_T?usp=sharing |                                                   |
+------------------+---------------------------------------------------+
| http             | Datasets of the human immune experiemnts          |
| s://drive.google |                                                   |
| .com/drive/folde |                                                   |
| rs/1OGMWxR7qTWd_ |                                                   |
| p21d57EyNWv5X48B |                                                   |
| NN0M?usp=sharing |                                                   |
+------------------+---------------------------------------------------+

If you have any questions, please contact: wanhui1997@pku.edu.cn
