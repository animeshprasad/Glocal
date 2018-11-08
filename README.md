# GAT
Graph Attention Networks


## Overview
Here we provide the implementation of a Graph Attention Network (GAT) layer in TensorFlow, along with a minimal execution example (on the KE dataset). The repository is organised as follows:
- `data/` contains the necessary dataset files for KE;
- `models/` contains the implementation of the GAT network (`gat.py`);
- `pre_trained/` contains a pre-trained KE model ;
- `utils/` contains:
    * an implementation of an attention head, along with an experimental sparse version (`layers.py`);
    * preprocessing subroutines (`process.py`);
    * preprocessing utilities for the PPI benchmark (`process_ppi.py`).

Finally, `execute_keyphrase_extraction.py` puts all of the above together and may be used to execute a full training run on KE.



## Dependencies

The script has been tested running under Python 3.5.2, with the following packages installed (along with their dependencies):

- `numpy==1.14.1`
- `scipy==1.0.0`
- `networkx==2.1`
- `tensorflow-gpu==1.6.0`

In addition, CUDA 9.0 and cuDNN 7 have been used.

## Reference
The base GAT model is taken from following manuscript:

```
@article{
  velickovic2018graph,
  title="{Graph Attention Networks}",
  author={Veli{\v{c}}kovi{\'{c}}, Petar and Cucurull, Guillem and Casanova, Arantxa and Romero, Adriana and Li{\`{o}}, Pietro and Bengio, Yoshua},
  journal={International Conference on Learning Representations},
  year={2018},
  url={https://openreview.net/forum?id=rJXMpikCZ},
  note={accepted as poster},
}
```

## License
MIT
