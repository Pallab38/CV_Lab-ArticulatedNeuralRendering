# cvgLab_anr

## Articulated Neural Rendering [[1]](#1).
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
#### Data Preparation
Following mesh normals are created to concate with the output of R_1 to be used as the input of R_2.
![Mesh Normals](/images/normal_all.jpg) <br>


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
