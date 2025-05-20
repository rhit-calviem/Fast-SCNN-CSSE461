# Instructions for Project

## Project Summary
This project implements and evaluates **Fast-SCNN**, a real-time semantic segmentation network, using the [Cityscapes dataset](https://www.cityscapes-dataset.com/). We retrained the model for 50 epochs, compared it to the original 160-epoch model, and tested both models on real-world images including Rose-Hulman campus scenes.

---

## Required Environment and Packages

We recommend using **Anaconda** for managing dependencies. Here are the necessary packages:

### Python version
- Python 3.8 or higher

### Conda setup
```bash
conda create -n fast-scnn python=3.8
conda activate fast-scnn
```

### Libraries needed
- os
- argparse
- torch
- PIL
- torchvision
- data_loader
- time
- shutil
- numpy
- math
- threading

### Changes made to code:

#### Train.py
Inside save_checkpoint():
`shutil.copyfile(filename, best_filename)`
This line tries to copy the saved file to another file if it's the best. But filename is just a name (e.g., fast_scnn_citys.pth), not a full path, so it will fail unless you’re in the save folder when running.
The variable save_path existed with a predefined path to the weights folder, so we switched the line of code to:
`shutil.copyfile(save_path, best_filename)`

#### Demo.py
The original code did allow to specify a custom .pth model: it always loaded the default pretrained model using:
`model = get_fast_scnn(args.dataset, pretrained=True, root=args.weights_folder, map_cpu=args.cpu).to(device)`
To fix this we made these two changes:
1. Inside parse_args(), we added a parser argument: <br>
`parser.add_argument('--resume', type=str, required=True, help='Path to the .pth model weights')`
2. inside demo(), we changed the boolean for pretrained to False: <br>
`model = get_fast_scnn(args.dataset, pretrained=False, root=args.weights_folder, map_cpu=args.cpu).to(device)`. <br>
After adding more code to take the .pth filed specified in the parser, this allowed us to use the model specified rather than always using the original weights.
<br>
We also had to make a couple of other changes to the demo() to make sure any image could be used in the demo. This was basic image preprocessing and resizing.

#### Eval.py
The original code also did not allow to specify a custom .pth model in the same exact way as demo.py. The code imported the parse_args() from train which contained the correct parser `parser.add_argument('--resume', type=str, default=None, help='put the path to resuming file if needed')`, so the only change needed was to set the boolean to false in `model = get_fast_scnn(args.dataset, pretrained=False, root=args.weights_folder, map_cpu=args.cpu).to(device)`.

---
## How To Run
Below we explain how to run the code to train, eval, and demo a model

### Train
The code to train is fully provided and requires no change. The dataets are too big and are not in Github; the authors suggest downloading the data here: [cityscapes](https://www.cityscapes-dataset.com/) from [here](https://www.cityscapes-dataset.com/downloads/). Specifically these two datasets:
1. [leftImg8bit_trainvaltest.zip(11GB)](https://www.cityscapes-dataset.com/file-handling/?packageID=4)
2. [gtFine_trainvaltest(241MB)](https://www.cityscapes-dataset.com/file-handling/?packageID=1).

Once downloaded, unzip and save the datasets into a new directory: `./datasets/citys`

To train run this code: <br>
`python train.py --model fast_scnn --dataset citys --epochs 50
`
<br>
The saved final .pth file for the weights of the model will be saved in the /weights directory. To specify a name for the file, change the `filename` in the save_checkpoint() method in train.py. We trained the model with 50 epochs, but you can specify any number of epochs that you want, the default epoch number is 160.

### Eval
The code to evaluate is fully provided and should not require any change. Run this line of code where you will need to specify your .pth filepath after --resume: <br>
`python eval.py --model fast_scnn --dataset citys --resume ./weights/fast_scnn_retrained.pth`

### Demo
The code to demo a model on any image is provided, same with all image processing so images of all sizes and shapes should be able to run. We have saved in the ./datasets directory the images we demoed our models on, so feel free to use these to check results. To run the demo, run this code: <br>
`python demo.py \--input-pic ./png/test_input.png \--resume ./weights/fast_scnn_retrained.pth \--outdir ./test_result_custom`
<br>
You will need to update the correct path to the picture and the model, the output directory is already given to you. Once the demo is finished, the outputted segmented image will be saved in the output directory.

---

## Sample Outputs with our Trained Model

1. An image from the cityscape dataset
   



 
