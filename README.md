# Computer Vision Lab: MA-INF:2307 

## Articulated Neural Rendering for Virtual Avatars [[1]](#1).
![Network Pipeline](/images/pipe00.jpg) <br>
Given a 3D body mesh, ANR generates a detailed avatar of that human body mesh using learned neural texture of the person. The 16 channel neural texture is rendered onto the image space by a weak perspective projection using rasterized IUV images of the mesh. The Neural renderer of our method is divided into two parts. First stage, R_1 converts the neural texture into another refined latent representation. Then the output of R_1 is concatenated with normal to be used as the input of second stage, R_2. R_2 produces an RGB rendering and a foreground mask.


### Run on Replik 
To run the **main.py** file on replik: 
1. Initialize the Replik project (``` replik init ```)
2. Include path (```/home/group-cvg/datasets/easymocap```) for data in **.replik/paths.json**. 
3. Run following from the Replik project directory:
```
replik schedule --script="main.py"
```


### Dataset
We use Lightstage data of the EasyMocap Dataset [[2]](#2) by ZJU. We are only using three videos where 3 subject performing warmup and punching. 

#### Structure of the dataset directory
![dir_tree](/images/dir_tree.png) <br>

#### Data Preparation
##### 1. First we extract the smpl mesh. Following gif shows extracted mesh on top of original frames for a given subject. <br>
<img src="/images/fitted_smpl_377.gif" width="480" height="480"><br>
##### 2. Following mesh normals are created to concate with the output of R_1 to be used as the input of R_2. 
![Mesh Normals](/images/normal_all.jpg) <br>


### Training 
+ Input: Around 44,000 frames with image resolution 256x 256 for all the three subjects.
+ Output: Rendered RGB and a foreground mask.
+ Loss: L_1 for rendererd RGB and Binary Cross Entropy for mask.
+ Optimizer: Adam optimizer with learning rate 0.002.
+ GPU: GTX 1080.
+ Runtime: Around 85 hours.


### Result 
#### Generated RGB and foreground mask for the subject performing warmup.
![377](/images/377_predictions.jpg) <br>
#### Generated RGB and foreground mask for the subject performing punching 01.
![386](/images/386_predictions.jpg) <br>
#### Generated RGB and foreground mask for the subject performing punching 02.
![387](/images/387_predictions.jpg) <br>



## References
<a id="1">[1]</a> 
Raj, Amit, Julian Tanke, James Hays, Minh Vo, Carsten Stoll, and Christoph Lassner. "ANR: Articulated Neural Rendering for Virtual Avatars." In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 3722-3731. 2021. <br/>
<a id = "2">[2] </a>
Peng, Sida, Yuanqing Zhang, Yinghao Xu, Qianqian Wang, Qing Shuai, Hujun Bao, and Xiaowei Zhou. "Neural body: Implicit neural representations with structured latent codes for novel view synthesis of dynamic humans." In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 9054-9063. 2021.
