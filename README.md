# ConSinGAN

todo give some example images from paper here

# Installation
todo give requirements.txt

```
pip install -r requirements.txt
```

# Unconditional Generation
To train a model with the default parameters from our paper run:

```
python main_train.py --gpu 0 --train_mode generation --input_name Images/Generation/angkorwat.jpg
```

To affect sample diversity and image quality we recomment playing around with the learning rate scaling (default is `0.1`) and the number of trained stages (default is `6`).
This can be especially helpful if the images are more complex (use a higher learning rate scaling) or you want to train on images with higher resolution (use more stages).
For example, increasing the learning rate scaling will mean that lower stages are trained with a higher learning rate and can, therefore, learn a more faithful model of the original image.
For example, observe the difference in generated images of the Colusseum if the model is trained with a learning rate scale of `0.1` or `0.5`:
todo inser example image here

```
python main_train.py --gpu 0 --train_mode generation --input_name Images/Generation/colusseum.jpg --lr_scale 0.5
```

Training on more stages can help with images that exhibit a large global structure that should stay the same:

```
python main_train.py --gpu 0 --train_mode generation --input_name Images/Generation/colusseum.jpg --train_stages 7
```


to sample:

```
todo
```

# Unconditional Generation (Arbitrary Sizes)
todo

# Harmonization
todo

# Editing
todo

# Additional Data
todo upload images of user studies and training images for user studies
