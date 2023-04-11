## Preparation
In `defaults.py`, modify `_C.BASE_DIRECTORY` in line 5 to the root directory where you intend to save the results.

## Training
To train the baseline source-only model, execute `python train.py --gpu GPU --config-file config.yaml`.
After each epoch, the model will be evaluated on both source and target validation data.
A npy-array `log.npy` of shape `num_epochs x 3` will be saved with the training loss in the first, and the validation Dice scores and the second and third column.

## Inference
To perform inference with a pretrained model, set the pathes in ll. 8-10 correctly and simply execute `python predict.py`.
This will save the predictions to the specified `out_path`.