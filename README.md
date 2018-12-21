## Agent-DOoM

<img src="https://github.com/Puneet2000/Agent-DOoM/blob/master/pic.png" width="500">

This is a RL Agent trained to play **Doom Deadly Corridor**. The Agent is trained using a [Double Deep Q-Learning](https://arxiv.org/pdf/1509.06461.pdf) with [Duel Architecture](https://arxiv.org/pdf/1511.06581.pdf) and [Prioritized Experience Replay](https://arxiv.org/pdf/1511.05952.pdf).

**Requirements**
- [VizDoom (Doom's Enviornment)](https://github.com/mwydmuch/ViZDoom/blob/master/doc/Building.md)
- PyTorch
- OpenCV

**Contents**
- ```utils.py``` : Prioritized Memory Replay , Sum Trees  , Preprocessing classes and fucntions.
- ```model.py``` : Dueling Deep Q Network
- ```doom.py``` : Training and Testing Agent.
- ```dqnet.py``` : weights after 1 hour training on GPU
- ```deadly_corridor.cfg``` : Enviornment Configuration File
- ```out.avi``` : Output video (Agent learnt to escape from enemies)

**Usage**
- To Train from starting : ```python3 doom.py --train=True --gpu=True --save_weights=True```
- Continue Training : ```python3 doom.py --weights=./dqnet.pt --train=True --gpu=True --save_weights=True```

**Flags**

| __Flag Name__ | __Description__ | __Default__ |
|-------------|------------|------------|
| ```--learning_rate``` | Learning rate for DQN | 0.025 |
| ```--episodes```  | Number of episodes for training | 5000 |
| ```--steps```  | Maximum steps per episode | 5000 |
| ```--tau```  | Time to refil target network weights | 3000 |
| ```--explore_stop```  | Stopping exploration probability | 0.01 |
| ```--decay_rate```  | Decay rate for exploration probability | 0.00005 |
| ```--discount_factor```  | Discounting factor for returns | 0.95 |
| ```--pretrain_length``` | number of iterations to fill replay memory | 10000 |
| ```--mem_size```  | Size of the replay memory | 10000 |
| ```--train```  | want to train or not | False |
| ```--batch_size```  | Batch size to sample from replay memory | 64 |
| ```--gpu```  | Use GPU | False|
| ```--weights``` | load pretrained weights | None |
| ```--save_weights```  | save the DQN weights | False |
| ```--graph```  | Draw summary graph | False |
| ```--test``` | Generate test video | False |

**Note 1 : Default Paramemters are used for training**

**Note 2: This Agent is trained only for 1 Hour on GPU and it has learned to escape from two enemies . Further training is needed to reach optimal policy**
