# 1. SBS_CLUSTERING
The SBS_CLUSTERING package is used to cluster somatic SBS mutation sequences.

## 1.1 Data available 

All the data was downloaded from [The Cancer Genome Atlas Program Database](https://www.cancer.gov/about-nci/organization/ccg/research/structural-genomics/tcga). Training and analysis of clinical data involved 2 kinds of files: .maf files contains SBS data and .xml files contains clinical data. 

## 1.2 Sample Data

Folder: `/SBS_CLUSTER/SAMPLE_DATA`

`./mut_info_XX.csv`:  the format of mutation information used for training, extracted from `.maf` files.

`./clinical_info.csv` : the format of patient information used for analysis, extracted from `.xml` file.

## 1.3 Data Pretreat

file: `/SBS_CLUSTER/dataloader.py`

Mutation bases were got from .maf file. Reference sequences were got from Genome Reference Consortium Human Build 38 (GRCh38) according to the mutation position and number of flanking bases needed to include in the training data. 

Use `dataloader.py` in the pretreating of training data.

>  Function: `GetWholeSequence(seq_path)`
>
>  Return variable contains the whole sequences for further analysis. 
>
>  Needed to install `pyfaidx` to read `.fa` file of whole gene sequence:
>
>  ```bash
>  pip install pyfaidx
>  ```
>
>  - seq_path: path of the `.fa` file.

>Function: `_get_mut_dict()`:
>
>Return a dictionary of base and its corresponding vector.

>Function `_get_patient_id()`:
>
>Return patient id according to 'Tumor_Sample_Barcode'.

> Function `ReadMafData(maf_file)`:
>
> Extract used items in a .maf file and return a numpy array
>
> - maf_file: path of .maf file

It is suggested to extract mutation information from .maf files and assign them into different .csv files randomly, in order to call data easily and avoid overfitting.

> Function `GetMutSeq(geneseq, mutpos, extend, bidirct)`
>
> Return flanking sequences of mutation base.
>
> - geneseq: whole reference sequence got by `GetWholeSequence`
> - mutpos: list of information of one mutation base in order of ['Chromosome', 'Start_Position', 'Reference_Allele', 'Tumor_Seq_Allele1', 'Tumor_Seq_Allele2'].
> - extend: length of sequence in each side of mutation base needed to included in training data.
> - bidirct: whether combine matrix of sequences of both direction together, if `bidirct == True`, return 1 combined matrix in horizontal direction, if `bidirct == False`, return 2 matrix of sequence on each side of mutation base. i.e. (1) `pre_list, post_list = GetMutSeq(geneseq=wholesequence, mutpos=['chr3', 300004, 'T', 'T', 'A'], extend=30, bidirct=False)`; (2) `combined_list = GetMutSeq(geneseq=wholesequence, mutpos=['chr3', 300004, 'T', 'T', 'A'], extend=30, bidirct=True)`

> Function `GetBatchSample(file_name, batchsize)`:
>
> Process .csv file into training data.
>
> Return a turple contains 2 numpy array used for training. The first array contains most of the samples in the file, The second list is the left samples that can't be made up into an entire batch of training data.
>
> - file_name: the .csv file contains mutation information.
> - batchsize: number of samples in each batch of training data.

## 1.4 Model building

file: `/SBS_CLUSTER/model.py`

LSTM-SOM model contains 2 parts: using LSTM to extract feature of mutation sequences, and use SOM for clustering of feature vector. `pytorch` is used in the building and training of LSTM-SOM model.

> Class `LSTMGene(torch.nn.Module)`:
>
> Build the LSTM model. Contains several hidden layers and one full-connection layer. 
>
> Function `__init__(self, input_size: int, hidden_size: int, output_size: int, n_layers: int)`:
>
> - input_size: dimension of each training data.
> - hidden_size: unit in each hidden layer.
> - output_size: dimension of output extracted feature vector.
> - n_layers: number of hidden layers. 
>
> Function `forward(self, x)`:
>
> Flow of input data in LSTM. Returen feature vector of one batch. 
>
> - x: input batch data, in shape of [length of sequence, batch size, dimension of each vector]

>class `SOMGene()`:
>
>Build the SOM model of LSTM-SOM model. 
>
>Function `__init__(self, weight, target_lr, sigma)`:
>
>- weight: units in competition layer, a numpy array in shape of (unmber of units, dimension of each input data).
>- target_lr: learning rate of SOM.
>- sigma: parameter in decay function.
>
>Function `Decay(self, distance)`:
>
>Decay function of SOM, determines the degree of attenuation with increasing distance. attenuation decrease with the increase of `sigma`.
>
>- distance: distance between current unit and target unit
>
>Function `CalusSOM(self, batch_sample)`:
>
>SOM process. Return variation of each batch of sample, in the same shape with input data. After each iter of SOM, value of units in competition layer will be refreshed.
>
>- batch_sample: input data of SOM.

## 1.5 Train

file: `/SBS_CLUSTER/train.py`

Train process of LSTM-SOM.

Constant `LOG_DICT`: Initalize the dictionary, further stored as the log file.

Constant `GRADE`: read from argparse '-g'. Because there are multiple rounds of training. And one round of training will use the data clustered by previous round of training. So the parameter `GRADE` contains information about the current round and which data to use. For example, if `GRADE` = 11000, it means round 2 of training, and use the 'class 1' data in round 1 training. 

i.e. `python traing.py -g 21000`

The trained model, log data, and sample graph will be stored in the corresponding folder defined in `cfg.py`. Model and log data will be saved each half the number of batches in each file,  and figure will be saved after every iter of training. Three of the eight dimensions are displayed using a spatial rectangular coordinate system

Note: if CPU version of pytorch is used, corresponding code should be modified.

## 1.6 Test

file: `/SBS_CLUSTER/test.py`

Test process and output result. The final result is the output of LSTM.

Constant `MODEL`: the path of trained model used for testing.

Constant `FILE_LIST`: the `.npy` file that store the numpy array of file list.

Constant `LIST_FROM`: path of folder contains the training/testing data.

Constant `CLASS_COL`: for each sample, the cluster result is a 1 × 8 vector. In most cases, in each dimension of output, a clear separation can be observed synchronized between different dimension. 

Constant `CLASS_LINE`: Use logistic regression to get the suitable boundary between different classes. This variable is used to difine the boundary between classes. At the same time, a upper limit value is required. For example, if there are 2 classes, values in class 2 is larger than class 1, the 2 classes is divided by 0, no value in class 2 is larger than 2, then `CLASS_LINE` can be defined as [3, 0].

Constant `RETRAIN`: if `RETRAIN = True`, clear the excited 'class' value, and give the new test result.

Constant `GRADE`: same will `GRADE` in `train.py`

argparse `-c`: whether fill the 'class' column with the class result. if `-c 0`, do not fill.

In the first round of training, a column named 'class' will be added to the file. If not the first round of training, the updated class result will also be filled in 'class' column. If use the data clustered as '1' in round 1 and '2' in round 2 in the 3rd round of training, use command :

```bash
python test.py -c 1 -g 31200
```

If this sample is clustered as '1' in 3rd round, the final class result is '1210'.

## 1.7 Configuration

file: `/SBS_CLUSTER/cfg.py`

There are some parameters in training and testing process, these parameters are written in `cfg.py`. 

Class `CFG`:

`CFG.BATCH_SIZE`: amount of samples in each batch. Default 100.

`CFG.WEIGHTSHAPE`: the shape of units in competition layer of SOM. The first element is the number of units in competition layer, the second element is the dimension of units in competition layer. Default (200, 8).

`CFG.EXTEND_NUM`: the length of flanking sequence involved in training data. Default 50.

`CFG.LSTMINPUTSIZE`: dimension of input vector. Default 8.

`CFG.LSTMHIDSIZE`: number of units in hidden layers of LSTM. Default 64.

`CFG.LSTMOUTPUTSIZE`: dimension of output vector of LSTM full connected layer. Default 8.

`CFG.LSTM_NUM_LAYERS`: number of hidden layers of LSTM. Default 1.

`CFG.LSTMLR`: Learning rate of LSTM. Default 0.001.

`CFG.SOMTARLR`: Learning rate of SOM. Default 0.001.

`CFG.SIGMA`: sigma of Nighborhood function in SOM. Default 0.4.

`CFG.PUSHTHRESHOLD`: precent of units move close to target. Default 0.05.

`CFG.LSTMITER`: times of LSTM in each iter of SOM. Default 2.

`CFG.BATCH_ITER`: times of SOM in each iter of training. Default 2.

`CFG.LOSS_PRINT`: print loss of LSTM every `CFG.LOSS_PRINT` iters of training. Default 2.

`CFG.WHOLE_SEQ_PATH`: path of `.fa` file saved whole gene sequences.

`CFG.MUT_INFO_PATH`: path of folder contains `.csv` file of mutation data.

`CFG.MODEL_PATH`: path that store the output `.tar` file of model infomation.

`CFG.LOG_PATH`: path that store the output `.json` file of training log.

`CFG.FIG_PATH`: path that store the output `.jpg` files of LSTM output in training process. Default 3 of 8 deminsions is shown in space rectangular coordinate system.

`CFG.PRED_INDEX`: the list of columns of output result that wish to add to the `.csv` file , number of elements in the list should be equal to the dimension of output vector.



