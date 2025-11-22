# 流体力学期中报告
基于 FEniCSx 的二维顶盖驱动方腔流数值模拟与分析
![image](https://github.com/dazhizhao/Lid-Driven-Cavity-Simulation/blob/main/lid_gif.gif)
# 环境配置
本报告使用FEniCSx 0.1进行仿真，区别于老版本FEniCS（已于2019年停止更新），首先配置环境：
```
conda create -n fenicsx-env
conda activate fenicsx-env
conda install -c conda-forge fenics-dolfinx mpich pyvista
```
同时安装Python tqdm库，运行下面安装：
```
conda activate fenicsx-env
pip install tqdm
```
# 仿真与结果保存
需要确保一直在FEniCSx环境下进行操作，开始仿真运行下面代码：
```
conda activate fenicsx-env
python lid_driven_cavity.py
```
仿真结果会以`.xdmf`和二进制`.h5`文件保存，后期可视化使用`Paraview`打开`.xdmf`即可。
