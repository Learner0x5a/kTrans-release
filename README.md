# kTrans: Knowledge-Aware Transformer for Binary Code Embedding

This repo is the official code of **kTrans: Knowledge-Aware Transformer for Binary Code Embedding**.

![Illustrating the performance on BSCD of kTrans](/figures/poolsizecompare.png)

## Requirements
 - NVIDIA GPU
 - CUDA 11.7
 - IDA Pro 7.6 (Linux)
 - pytorch 2.0
 - pytorch Lightning 2.0
 - transformers 4.27.1
 - captone 4.0.2

## Download link for the pretrained kTrans model

https://drive.google.com/file/d/1SvS4tO-UVvZDvdubl9jVgUKJtjzdxa7V/view?usp=sharing

```bash
$ md5sum ktrans-110M-epoch-2.ckpt
6628c8a5e0bb9689bce0d5fe10bde734  ktrans-110M-epoch-2.ckpt
``` 

## Usage to generate embedding from a given binary program


## 1. Generate instructions along with metadata with IDA
This step generates a pickle file containing a list of [instruction addresses,disassembly,instruction,operand type, operand r/w status, eflags]

e.g. `/root/idapro-7.6/idat64 -L"ida.log" -A -S"./insn_rw.py ./ida_outputs" ./demo_bin/ls`

```bash
# Note: Remember to change IDA's python interpreter to the system's python interpreter (the one equipped with PyTorch, Capstone, etc.)
/path/to/ida -L"ida.log" -A -S"./insn_rw.py output_dir" /path/to/target_binary
```

## 2. Generate embedding with kTrans
This step generates embedding with kTrans and stores the output into a pickle file (numpy.ndarray in shape (num_functions, embedding_dim))

e.g. `python3 ktrans-gen-emb.py -i ./ida_outputs -o ./saved_emb -m ./ktrans-110M-epoch-2.ckpt -n 32 -bs 128`

```bash
python3 ktrans-gen-emb.py -i /path/to/pickles_gen_by_ida -o /path/to/saved_embeddings -m /path/to/kTrans_model -n num_workers_for_dataloader -bs inference_batch_size
```


## Acknowledgement
This project is not possible without multiple great open-sourced code bases. We list some notable examples below.

* [transformers](https://github.com/huggingface/transformers)
* [lightning](https://github.com/lightning-ai/lightning)

## Bibtex
If this work or BinaryCorp dataset are helpful for your research, please consider citing the following BibTeX entries.

```
@article{zhu2023ktrans,
  title={kTrans: Knowledge-Aware Transformer for Binary Code Embedding},
  author={Zhu, Wenyu and Wang, Hao and Zhou, Yuchen and Wang, Jiaming and Sha, Zihan and Gao, Zeyu and Zhang, Chao},
  journal={arXiv preprint arXiv:2308.12659},
  year={2023}
}

@inproceedings{10.1145/3533767.3534367,
author = {Wang, Hao and Qu, Wenjie and Katz, Gilad and Zhu, Wenyu and Gao, Zeyu and Qiu, Han and Zhuge, Jianwei and Zhang, Chao},
title = {JTrans: Jump-Aware Transformer for Binary Code Similarity Detection},
year = {2022},
isbn = {9781450393799},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3533767.3534367},
doi = {10.1145/3533767.3534367},
abstract = {Binary code similarity detection (BCSD) has important applications in various fields such as vulnerabilities detection, software component analysis, and reverse engineering. Recent studies have shown that deep neural networks (DNNs) can comprehend instructions or control-flow graphs (CFG) of binary code and support BCSD. In this study, we propose a novel Transformer-based approach, namely jTrans, to learn representations of binary code. It is the first solution that embeds control flow information of binary code into Transformer-based language models, by using a novel jump-aware representation of the analyzed binaries and a newly-designed pre-training task. Additionally, we release to the community a newly-created large dataset of binaries, BinaryCorp, which is the most diverse to date. Evaluation results show that jTrans outperforms state-of-the-art (SOTA) approaches on this more challenging dataset by 30.5% (i.e., from 32.0% to 62.5%). In a real-world task of known vulnerability searching, jTrans achieves a recall that is 2X higher than existing SOTA baselines.},
booktitle = {Proceedings of the 31st ACM SIGSOFT International Symposium on Software Testing and Analysis},
pages = {1–13},
numpages = {13},
keywords = {Binary Analysis, Similarity Detection, Neural Networks, Datasets},
location = {Virtual, South Korea},
series = {ISSTA 2022}
}
```

