# TransformerVLN

Our project use Matterport3D Simulator and Room-to-Room (R2R) Dataset [(Anderson et al., 2018)](https://arxiv.org/abs/1711.07280), so before running our code, you need to build an environment. 



## If you run our code in a colab environment

We are using Colab Pro server to set up the environment and train our model. Here you can directly use the following colab notebook to build the environment and run our code if you have enough RAM memory (you need to have colab pro and enough storage in google drive).

Colab notebook: [transformer_vln](https://colab.research.google.com/drive/164ULWvQg_Bricrw95Z3E1XsozmpdjgRl?usp=sharing)

In the notebook you should first mount the Google Drive, then download source code, libraries, pre-computed image features, then start training. Be sure that you connect to GPU session and activate the "High-RAM" mode of Colab Pro to ensure there are enough RAM memory to run the code. 



## If you run our code on your own server

#### Environment Installation Instruction

#### Prerequisites

- Ubuntu >= 14.04
- Nvidia-driver with CUDA installed 
- C++ compiler with C++11 support
- [CMake](https://cmake.org/) >= 3.10
- [OpenCV](http://opencv.org/) >= 2.4 including 3.x
- [OpenGL](https://www.opengl.org/)
- [GLM](https://glm.g-truc.net/0.9.8/index.html)
- [Numpy](http://www.numpy.org/)



Clone our GitHub repository, make sure to clone with --recursive

```
git clone --recursive https://github.com/ameimeimei73/TransformerVLN.git
```

Download R2R dataset

```
bash ./tasks/R2R/data/download.sh
```

Download image features for environments

```
mkdir img_features
wget https://www.dropbox.com/s/o57kxh2mn5rkx4o/ResNet-152-imagenet.zip -P img_features/
cd img_features
unzip ResNet-152-imagenet.zip
cd ..
```

Python requirements

```
pip install -r tasks/R2R/requirements.txt
```

Install Matterport3D simulators

```
sudo apt-get install libjsoncpp-dev libepoxy-dev libglm-dev libosmesa6 libosmesa6-dev libglew-dev
pip install dgl-cu111 -f https://data.dgl.ai/wheels/repo.html
mkdir build
cd build
cmake -DOSMESA_RENDERING=ON ..
make -j8
```



#### Run code

**Baselines**

Our baselines are the sequence-to-sequence LSTM model and some simple agents from [Anderson et al., 2018](https://arxiv.org/abs/1711.07280), you can find their code in the [github repository](https://github.com/peteanderson80/Matterport3DSimulator/tree/master/tasks/R2R). 

- Seq2seq model: run train.py
- Random and shortest agent: run eval.py.



**Our approaches**

The implementation of our three approaches are based on the code in baseline, we reuse the code of the interaction with the simulator environment. 

- BERT + FCs model and BERT + LSTM model

  The code of these two approach is in `tasks/R2R/bert`. Our implementation of the models are in `model.py`, and the training process is in `train_bert_based.py` and `agent.py`.

  To train BERT + FCs model:

  ```
  python tasks/R2R/bert/train_bert_based.py BERT_FC
  ```

  To train BERT + LSTM model:

  ```
  python tasks/R2R/bert/train_bert_based.py BERT_LSTM
  ```

- Customized T5 model

  The code of this approach is in `tasks/R2R/t5`. Our implementation of the models are in `model.py`, and the training process is in `train_t5.py` and `agent.py`.

  To train Customized T5 model:

  ```
  python tasks/R2R/t5/train_t5.py
  ```