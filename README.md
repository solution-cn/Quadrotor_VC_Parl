# Quadrotor_VC_Parl
基于Parl框架：利用Model、Algorithm、Agent以及其内部已集成的算法DDPG复现rlschool内的Quadrotor速度控制任务。
===

才看到[PaddlePaddle /RLSchool /use body velocity as target in velocity_control task #31](https://github.com/PaddlePaddle/RLSchool/pull/31/commits/e94b5d8142f4e9bfbcea95f7849d97b37cb45c59)的修改说明：<br>
此处修改的是：将原来的velocity_target是根据global以初始姿态先全部一次性化成body再开始训练（未考虑到无人机姿态的时变性），现更改velocity_target为每次reward的时候才将global化为body进行reward计算。<br>（*注意：修改过后本模型中在外部（env内则不需要）求的reward也要根据实际情况进行修改）<br>
>目前模型尚未来得及进行训练替换，但本模型1、2在将obs中的body_v靠obs内其它状态参数应该足够在网络内部完成收敛。如果想加快收敛速度亦可以将obs中的body_v靠obs内其它状态参数预处理为global后进行输入。<br>
>具体方法可以参考rlschool进行global to body的方法进行反向操作。env的simulator（Class QuadrotorSim）内含参数self._coordination_converter_to_world。<br>
>未对这个Trick的有效性进行测试，换成Globle虽然让实际速度和目标速度在同一维度，但四电机也转换成Globle velocity，也是难度一定程度上大了的。<br>
（

---
在实践过程中发现不管如何调参或者调整Action的训练策略（4个电机单独控制抑或采用5个参数分别表示基础电压以及四个电机的Offset）都轻易能够将reward下调至-21左右。但在render观察实测效果发现这个任务并没有很好地完成，无人机只学会了“省电”（实际上是学会减少各个电机的电压变化）。<br> 
>个人经过对rlschool中Quadrotor环境相关的代码进行研究，发现内部设置的reward是由“寿命”（指续航能力）、“目标”、“状态”组成。其中关于寿命的指标是一致的，经过测试，是指无人机电机电压变化的快慢，突变得越快，值越高，最低由内部参数healthy_reward控制，默认为1，取前者乘以系数dt(0.01)和healthy_reward最小值，再取  **负** 作为关于寿命的reward部分。<br> 
>而对于目标部分，只采用了实际速度与目标速度的三维相差的绝对值之和，而且乘以系数-0.001。无人机速度三维里最多-10到10，这意味着即使是相差得最大，60*-0.001=-0.06的reward，与电机突变，四个电机从0.1V到15V造成的-22.7(2270+)最后取-1,相比之下孰轻孰重无人机当然轻松学得会。<br> 
>实际上只需要将每次的动作(action)设置成常量不变就能够取得-21左右的reward。<br> 
>也就是rlschool内Quadrotor环境中velocity_control任务中原设的reward不能指引Agent完成工作的目标。因此必须重新设计能够表示速度控制这一任务的reward供Agent参考训练。<br> 

观察任务要求，采样一次目标速度，并通过粗略运算（30fps）计算加速度。<br> 
>![](https://github.com/solution-cn/pic/blob/master/V.png)<br>
>![](https://github.com/solution-cn/pic/blob/master/A.png)<br>

---
显然面对这种有急加速需求的任务，不能再直接采用原来的寿命计算方式。但本人对于无人机电机的研究不甚了解，无法定义一个比较好的进行替代。<br>
据此考虑了几种方式：<br> 
1.采用原reward与xyz实际速度与目标速度的方差之和相结合：<br> 
reward += k * ((vx-t_vx)^2 + (vy-t_vy)^2 + (vz-t_vz)^2)<br> 
k取-0.01<br> 
2.剔除原reward，计算xyz实际速度与目标速度的方差之和与xyz实际速度与目标速度的差的绝对值之和（考虑到速度差接近1以内时后者比前者更适合作为reward）：<br> 
reward = k *  ((vx-t_vx)^2 + (vy-t_vy)^2 + (vz-t_vz)^2 + |vx-t_vx| + |vy-t_vy| + |vz-t_vz|)<br> 
k取-0.1<br> 
最后均在测试时计算该环境任务内计算的原reward进行观察。<br> 
(v和t_v分别表示实际速度与目标速度)

---
第一种经过训练后取得（C_R表示修正后的reward,D_R表示环境内原reward）[4个电机独立工作]<br> 
见Method01.py<br> 
C_R:-1080.4929236394496 , D_R:-762.3663688514929<br> 
已经明显能看出橙线（实际速度）会在大幅偏离黄线（目标速度）时的调整动作。<br> 
>![](https://github.com/solution-cn/pic/blob/master/123.gif)  <br> 

第二种经过训练后取得（C_R表示修正后的reward,D_R表示环境内原reward）[4个电机独立工作]<br>
见Method02.py<br> 
C_R:-1093.8570576218024 , D_R:-958.0931151165605<br>
在去除寿命指标后调整动作更加迅速<br>
>![](https://github.com/solution-cn/pic/blob/master/L3.gif)  <br> 

最后附上velocity.py文件，以证明电机保持0.1电压也可以在该环境中以原reward计算方式取得平均-20左右的test_reward。<br>
见 velocity.py<br>

---
