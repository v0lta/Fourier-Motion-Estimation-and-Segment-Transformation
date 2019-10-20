# masterproject_videopred
Master Thesis: Recurrent Neural Networks (RNNs) for Future Frame Prediction in Video Sequences
## Dataset
MovingMNIST++: more details can be seen from [HKO-7](https://github.com/sxjscience/HKO-7).

Generate the dataset
```bash
python generatedata.py --seqlen=16 --digit_num=3 --num=2000
```
The example above generates 2000 sequences, each is 16 frames long, each frame has 3 digits within it.
## Running the experiments
We are using python 3.5 and Tensorflow 1.4
```bash
python train_main.py -bs=2 -e=10 -lr=0.0001 --seqlen=4 --pred_num=2 --digit_num=2 --cell=TrajGRU --network=tcn --encoder=2 -l=2 --predictor
```
We have implemented ConvGRU, TrajGRU, FlowGRU, FourierGRU and StridedConvGRU. The network architecture can be set to be temporal compressed through *network* argument. Our model can support both the prediction and the reconstruction though in the paper we only talked about the prediction. More information about the arguments is in the train_main.py script.

## Tools
In the **helpers** directory, we provide scripts for analysing logs and for results visualization and evaluation.
