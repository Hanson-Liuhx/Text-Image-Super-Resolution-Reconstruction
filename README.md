# Text-Image-Super-Resolution-Reconstruction
CS172 Final project: Text Image Super-Resolution Reconstruction

Operation Manual for Codes

Dataset (ICDAR2015 Competition Dataset):

  You can download this dataset from here: https://github.com/piclem/ICDAR2015-TextSR/blob/master/ICDAR2015-TextSR-dataset.zip?raw=true

Prerequisites: Pytorch, Scipy, Numpy, Tesseract OCR, Pytesseract

------------------------------------------------------------------------------------------

Train:

  To train a new model, change the directories path at:
  
     train.py: line 35, 36
     
  to the directory of the dataset.
  
  Then run
  
    python3 train.py
    
  Optional arguments:
  
    --num_epochs                    train epoch number [default value is 100]
    
  The output super resolution images are on training_results directory.
  
--------------------------------------------------------------
 
Test:

  To test a model, change the directories path at:
  
    test_benchmark.py: line 46
    
  to the directory of the test set.
  
  Then run
  
    python3 test_benchmark.py
    
  Optional arguments:
  
    --model_name                   generator model epoch name [default value is netG_epoch_4_100.pth]
    
    --ocr                          use OCR for validation [default value is 0]
    
  The output super resolution images are on benchmark_results directory.
  
