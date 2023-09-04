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
If this work or BinaryCorp dataset are helpful for your research, please consider citing the following BibTeX entry.

```
@article{zhu2023ktrans,
  title={kTrans: Knowledge-Aware Transformer for Binary Code Embedding},
  author={Zhu, Wenyu and Wang, Hao and Zhou, Yuchen and Wang, Jiaming and Sha, Zihan and Gao, Zeyu and Zhang, Chao},
  journal={arXiv preprint arXiv:2308.12659},
  year={2023}
}
```

