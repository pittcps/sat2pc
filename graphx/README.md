# GraphX

This script for the GraphX baseline is based on the original implementation provided [here](https://github.com/justanhduc/graphx-conv) with some modifications to add support for our dataset.

# Usage

To train the model you can run the following command:

```
python train.py --config ../configs/graphx.gin --data-dir ../datasets/
```

To test the model you can run the following command:

```
python test.py --config ../configs/graphx.gin --data-dir ../datasets/ --ckpt-path [location of the saved weights]
```
