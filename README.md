# FasterICA

PyTorch implementation of FasterICA. 

Implements the following algorithms:
- Batch PCA with optimistic whitening
    - D. Ross, J. Lim, R. Lin, M. Yang, Incremental Learning for Robust Visual 
    Tracking, International Journal of Computer Vision, Volume 77, Issue 1-3,
    pp. 125-141, May 2008. 
    - See https://www.cs.toronto.edu/~dross/ivt/RossLimLinYang_ijcv.pdf
- Improved Lie Group ICA
    - Plumbley, M. D. (2007, April). Geometry and manifolds for independent component analysis. In 2007 IEEE International Conference on Acoustics, Speech and Signal Processing-ICASSP'07 (Vol. 4, pp. IV-1397). 
    - See http://www.eecs.qmul.ac.uk/~markp/2007/Plumbley07-icassp.pdf


### Installation

1. For GIT:
```
git clone git@git.ios.htwg-konstanz.de:mof-applications/fasterica.git
python setup.py install|develop
```

2. For PIP
```
pip install git+https://git.ios.htwg-konstanz.de/mof-applications/fasterica.git
```

### Run example
```
python examples/sklearn_digits.py  
```

### Run tests

```
pytest tests
```
