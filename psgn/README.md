# PointSet

This script for the GraphX baseline is based on the implementation provided [here](https://github.com/ywcmaike/PSGN/tree/31fad8815b09b8434c597974824917d7eb9e355e) with some modifications to add support for our dataset.

# Usage

To train the model you can run the following command:

```
python train.py --data-dir ../datasets/
```

To test the model you can run the following command:

```
python test.py --data-dir ../datasets/ --ckpt-path [location of the saved weights]
```
