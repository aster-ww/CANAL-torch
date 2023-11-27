# CANAL: Continually adapting pre-trained language model to universal annotation of single-cell RNA-seq data
PyTorch implementation of CANAL,  a universal cell-type annotation tool that continuously fine-tunes a pretrained language model trained on a large amount of unlabeled scRNA-seq data, as new well-labeled data emerges.

<p align="center">
<img src="https://github.com/aster-ww/CANAL/blob/main/framework.jpg" width="1100" align="center">
</p>

# File Descriptions and data requirement

Cell-type annotation of single-cell RNA-seq (scRNA-seq) data is a hallmark of biomedical research and clinical application. Current annotation tools usually assume the simultaneous acquisition of well-annotated data, but without the ability to expand knowledge from new data. Yet, such tools are inconsistent with the continuous emergence of scRNA-seq data, calling for a continuous cell-type annotation model. In addition, by their powerful ability of information integration and model interpretability, transformer-based pre-trained language models have led to breakthroughs in single-cell biology research.  We herein propose a universal cell-type annotation tool, called CANAL, that continuously fine-tunes a pretrained language model trained on a large amount of unlabeled scRNA-seq data, as new well-labeled data emerges. For model inputs, we designed an experience replay schema that repeatedly reviews previous vital examples in current training stages via a dynamic example bank with fixed buffer size. This example bank is class-balanced and good at memorizing and consolidating cell-type-specific information, as well as patterns of rare cell types. For model outputs, we utilize representation knowledge distillation to regularize the divergence between previous and current models, resulting in the preservation of knowledge learned from past training stages. Moreover, our universal annotation framework considers new cell types during both the fine-tuning and testing stages. 


 # Hyper-parameters and recommended settings

>- lambda: default 0.1, the strength of representation distillation loss
>
>- L: default 1000, the size of example bank
>

# Data Availability

|Link|Description|
|----|-----------|
|https://drive.google.com/drive/folders/1BMf-N-k-3aCEY7CJvUcK9nZZ2UD7p3C0?usp=sharing| Datasets of the pancreas experiemnts|
|https://drive.google.com/drive/folders/1CaBySV_EFAPPrlpSevEewFds5cjJxC_T?usp=sharing| Datasets of the cross-tissue experiemnts |
|https://drive.google.com/drive/folders/1OGMWxR7qTWd_p21d57EyNWv5X48BNN0M?usp=sharing| Datasets of the human immune experiemnts |


If you have any questions, please contact: wanhui1997@pku.edu.cn
