### 切实可用的命令

```bash
sc create --name seg  --image "harbor.smoa.cc/public/xsemseg:v1.4" --gpu 1 --cmd "source /dataset/vsitongwu/.bashrc && cd /dataset/vsitongwu/LJC/TargetSeg/semseg && conda activate animate && sh tool/train.sh target pspnet50" --arch ampere --debug
--node node66
```

> 常用命令
>
> 1. 创建任务（debug）
>   sc create --name t --image "harbor.smoa.cc/public/xsemseg:v1.4" --gpu 1 --cmd "sleep 2h" --debug    ## 2080ti/v100
>   sc create --name t  --image "harbor.smoa.cc/public/xsemseg:v1.4" --gpu 2 --cmd "sleep 2h" --debug --arch ampere ## 3090
> 2. 创建任务
>   sc create --name d1 --image "harbor.smoa.cc/public/xsemseg:v1.4" --gpu 1 --cmd "YOURCMD"  --arch ampere
> 3. 手动指定某个节点： --node nodename
> 4. 查看当前任务中间输出
>   sc log taskname
> 5. 删除任务
>   sc delete taskname
> 6. 查看所有任务状态
>   sc list
> 7. 查看某个任务具体情况
>   sc describe taskname
> 8. 查看所有节点情况
>   sc node list
>

git 命令
```bash
git config user.name "ljc"
git config user.email "l_ai@ai.com"
git remote add origin git@github-ljc:jirange/segmentation-target.git
git config --list
git push origin master
```

```bash
框架准备 数据集
unzip images.zip -d ./images/TargetS
ln -s images/TargetS semseg/dataset/TargetS

unzip voc2012-20240612T151755Z-001.zip -d ./images/voc2012
ln -s images/VOCdevkit/VOC2012/ semseg/dataset/voc2012
```

------

### 报错分析

sh tool/test.sh voc2012 pspnet50

> Traceback (most recent call last):
>  File "/dataset/vsitongwu/LJC/TargetSeg/semseg/exp/voc2012/pspnet50/test.py", line 6, in <module>
>    import cv2
>  File "/dataset/vsitongwu/anaconda3/envs/animate/lib/python3.10/site-packages/cv2/__init__.py", line 181, in <module>
>    bootstrap()
>  File "/dataset/vsitongwu/anaconda3/envs/animate/lib/python3.10/site-packages/cv2/__init__.py", line 153, in bootstrap
>    native_module = importlib.import_module("cv2")
>  File "/dataset/vsitongwu/anaconda3/envs/animate/lib/python3.10/importlib/__init__.py", line 126, in import_module
>    return _bootstrap._gcd_import(name[level:], package, level)
> ImportError: libGL.so.1: cannot open shared object file: No such file or directory
>
> first-try: pip install opencv-python-headless    OK!

-----

```bash
sh tool/mytest.sh voc2012 pspnet50

sc create --name seg  --image "harbor.smoa.cc/public/xsemseg:v1.4" --gpu 1 --cmd "source /dataset/vsitongwu/.bashrc && cd /dataset/vsitongwu/LJC/TargetSeg/semseg && conda activate svd-simple && sh tool/mytest.sh voc2012 pspnet50" --debug --arch ampere
```

> [2024-06-13 06:06:56,829 INFO test.py line 248 24] Eval result: mIoU/mAcc/allAcc 0.0431/0.0466/0.9410.

voc2012数据集下载

> http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar

```bash
sh tool/train.sh target pspnet50

sh tool/mytrain.sh target pspnet50

sc create --name seg  --image "harbor.smoa.cc/public/xsemseg:v1.4" --gpu 2 --cmd "source /dataset/vsitongwu/.bashrc && cd /dataset/vsitongwu/LJC/TargetSeg/semseg && conda activate animate && sh tool/mytrain.sh target pspnet50" --debug --arch ampere
```

> - ModuleNotFoundError: No module named 'tensorboardX'
> - pip install tensorboardX

```bash
 sc create --name seg  --image "harbor.smoa.cc/public/xsemseg:v1.4" --gpu 2 --cmd "source /dataset/vsitongwu/.bashrc && cd /dataset/vsitongwu/LJC/TargetSeg/semseg && conda activate animate && sh tool/mytrain.sh target pspnet50" --arch ampere --node node66
```



> Traceback (most recent call last):
>   File "/dataset/vsitongwu/LJC/TargetSeg/semseg/exp/target/pspnet50/train.py", line 410, in <module>
>     main()
>   File "/dataset/vsitongwu/LJC/TargetSeg/semseg/exp/target/pspnet50/train.py", line 106, in main
>     mp.spawn(main_worker, nprocs=args.ngpus_per_node, args=(args.ngpus_per_node, args))
>   File "/dataset/vsitongwu/anaconda3/envs/animate/lib/python3.10/site-packages/torch/multiprocessing/spawn.py", line 239, in spawn
>     return start_processes(fn, args, nprocs, join, daemon, start_method='spawn')
>   File "/dataset/vsitongwu/anaconda3/envs/animate/lib/python3.10/site-packages/torch/multiprocessing/spawn.py", line 197, in start_processes
>     while not context.join():
>   File "/dataset/vsitongwu/anaconda3/envs/animate/lib/python3.10/site-packages/torch/multiprocessing/spawn.py", line 160, in join
>     raise ProcessRaisedException(msg, error_index, failed_process.pid)
> torch.multiprocessing.spawn.ProcessRaisedException: 
>
> -- Process 0 terminated with the following error:
> Traceback (most recent call last):
>   File "/dataset/vsitongwu/anaconda3/envs/animate/lib/python3.10/site-packages/torch/multiprocessing/spawn.py", line 69, in _wrap
>     fn(i, *args)
>   File "/dataset/vsitongwu/LJC/TargetSeg/semseg/exp/target/pspnet50/train.py", line 195, in main_worker
>     transform.RandScale([args.scale_min, args.scale_max]),
>   File "/dataset/vsitongwu/LJC/TargetSeg/semseg/util/transform.py", line 79, in __init__
>     assert (isinstance(scale, collections.Iterable) and len(scale) == 2)
> **AttributeError: module 'collections' has no attribute 'Iterable'**

> AttributeError: module 'collections' has no attribute 'Iterable' 表示 collections 模块中没有名为 Iterable 的属性。在Python 3.3及以后的版本中，collections 模块确实没有 Iterable 这个属性。在Python 2.7中，Iterable 是 collections 模块的一部分，但在Python 3中已经被移除。
>
> 应该**使用 collections.abc.Iterable 而不是 collections.Iterable**。collections.abc 是在Python 3中引入的，用于提供抽象基类（ABCs）

结果输出：

```bash
Totally 96 samples in train set.
Starting Checking image&label pair train list...
Checking image&label pair train list done!
Totally 96 samples in train set.
Starting Checking image&label pair train list...
Checking image&label pair train list done!
[2024-06-13 07:43:22,530 INFO train.py line 339 90] Train result at epoch [1/50]: mIoU/mAcc/allAcc 0.4484/0.4484/0.8968.
[2024-06-13 07:43:22,531 INFO train.py line 233 90] Saving checkpoint to: exp/target/pspnet50/model/train_epoch_1.pth
[2024-06-13 07:43:47,858 INFO train.py line 339 90] Train result at epoch [2/50]: mIoU/mAcc/allAcc 0.5000/0.5000/1.0000.
[2024-06-13 07:43:47,859 INFO train.py line 233 90] Saving checkpoint to: exp/target/pspnet50/model/train_epoch_2.pth
[2024-06-13 07:44:13,976 INFO train.py line 339 90] Train result at epoch [3/50]: mIoU/mAcc/allAcc 0.5000/0.5000/1.0000.
[2024-06-13 07:44:13,977 INFO train.py line 233 90] Saving checkpoint to: exp/target/pspnet50/model/train_epoch_3.pth
[2024-06-13 08:03:43,568 INFO train.py line 339 90] Train result at epoch [50/50]: mIoU/mAcc/allAcc 0.5000/0.5000/1.0000.
[2024-06-13 08:03:43,569 INFO train.py line 233 90] Saving checkpoint to: exp/target/pspnet50/model/train_epoch_50.pth
```



>
> Exception in thread Thread-1:
> Traceback (most recent call last):
>   File "/dataset/vsitongwu/anaconda3/envs/animate/lib/python3.10/threading.py", line 1016, in _bootstrap_inner
>     self.run()
>   File "/dataset/vsitongwu/anaconda3/envs/animate/lib/python3.10/site-packages/tensorboardX/event_file_writer.py", line 202, in run
>     data = self._queue.get(True, queue_wait_duration)
>   File "/dataset/vsitongwu/anaconda3/envs/animate/lib/python3.10/multiprocessing/queues.py", line 117, in get
>     res = self._recv_bytes()
>   File "/dataset/vsitongwu/anaconda3/envs/animate/lib/python3.10/multiprocessing/connection.py", line 212, in recv_bytes
>     self._check_closed()
>   File "/dataset/vsitongwu/anaconda3/envs/animate/lib/python3.10/multiprocessing/connection.py", line 136, in _check_closed
>     raise OSError("handle is closed")
> OSError: handle is closed

2*3090 20mins

> sh tool/mytest.sh target pspnet50
> sc create --name seg  --image "harbor.smoa.cc/public/xsemseg:v1.4" --gpu **1** --cmd "source /dataset/vsitongwu/.bashrc && cd /dataset/vsitongwu/LJC/TargetSeg/semseg && conda activate animate && sh tool/mytest.sh target pspnet50" --debug --arch ampere

> [2024-06-13 08:10:38,391 INFO test.py line 74 24] => creating model ...
> [2024-06-13 08:10:38,391 INFO test.py line 75 24] Classes: 2
> Totally 24 samples in val set.
> Starting Checking image&label pair val list...
> Checking image&label pair val list done!
> 省略
> [2024-06-13 08:10:42,023 INFO test.py line 111 24] => loading checkpoint 'exp/target/pspnet50/model/train_epoch_50.pth'
> [2024-06-13 08:10:43,896 INFO test.py line 114 24] => loaded checkpoint 'exp/target/pspnet50/model/train_epoch_50.pth'
> [2024-06-13 08:10:43,896 INFO test.py line 182 24] >>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>
> [2024-06-13 08:10:54,345 INFO test.py line 208 24] Test: [10/24] Data 0.026 (0.316) Batch 0.186 (1.045).
> [2024-06-13 08:10:56,265 INFO test.py line 208 24] Test: [20/24] Data 0.021 (0.171) Batch 0.197 (0.618).
> [2024-06-13 08:10:57,028 INFO test.py line 208 24] Test: [24/24] Data 0.026 (0.147) Batch 0.188 (0.547).
> [2024-06-13 08:10:57,147 INFO test.py line 223 24] <<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<
> [2024-06-13 08:10:57,173 INFO test.py line 240 24] Evaluating 1/24 on image 夜晚_ROI_2_20240521205622628-1975.png, accuracy 1.0000.
> 省略
> [2024-06-13 08:10:57,726 INFO test.py line 240 24] Evaluating 24/24 on image 晴天白天_ROI_1_20240522115828659-3721.png, accuracy 1.0000.
> [2024-06-13 08:10:57,726 INFO test.py line 248 24] Eval result: mIoU/mAcc/allAcc 0.5000/0.5000/1.0000.
> [2024-06-13 08:10:57,726 INFO test.py line 250 24] Class_0 result: iou/accuracy 1.0000/1.0000, name: background.
> [2024-06-13 08:10:57,726 INFO test.py line 250 24] Class_1 result: iou/accuracy 0.0000/0.0000, name: target.




> ../aten/src/ATen/native/cuda/NLLLoss2d.cu:103: nll_loss2d_forward_kernel: block: [3,0,0], thread: [226,0,0] Assertion `t >= 0 && t < n_classes` failed.
> ../aten/src/ATen/native/cuda/NLLLoss2d.cu:103: nll_loss2d_forward_kernel: block: [3,0,0], thread: [736,0,0] Assertion `t >= 0 && t < n_classes` failed.
> ../aten/src/ATen/native/cuda/NLLLoss2d.cu:103: nll_loss2d_forward_kernel: block: [3,0,0], thread: [737,0,0] Assertion `t >= 0 && t < n_classes` failed.
> ../aten/src/ATen/native/cuda/NLLLoss2d.cu:103: nll_loss2d_forward_kernel: block: [3,0,0], thread: [738,0,0] **Assertion `t >= 0 && t < n_classes` failed.**
> train_h train_w引起的   或者是transform ignore_label 随意变换引起的  **ignore_label 与label数值的范围需要对应**

> torch.multiprocessing.spawn.ProcessExitedException: process 1 terminated with signal SIGABRT
> /dataset/vsitongwu/anaconda3/envs/animate/lib/python3.10/multiprocessing/resource_tracker.py:224: UserWarning: resource_tracker: There appear to be 3 leaked semaphore objects to clean up at shutdown
>   warnings.warn('resource_tracker: **There appear to be %d '**
>   ignore_label 随意变换引起的    **ignore_label 与label数值的范围需要对应**

> 通过添加label/255 解决前二者的问题  感觉与内存泄漏和标注为忽略后占用存储少了有关。 不过还是不知道为什么 模模糊糊 隐隐约约删除忽略标签的同时 /255  不过如果只是/255 继续采用忽略标签呢， 而且有个小问题 只有background的macc啥的是正常的 target全0 还是怀疑和忽略标签有关

-----

### 测试结果

[2024-06-14 08:50:42,554 INFO test.py line 248 24] Eval result: mIoU/mAcc/allAcc **0.4831/0.4949/0.9898**.
[2024-06-14 08:50:42,554 INFO test.py line 250 24] Class_0 result: iou/accuracy 0.9661/0.9898, name: background.
[2024-06-14 08:50:42,554 INFO test.py line 250 24] Class_1 result: iou/accuracy 0.0000/0.0000, name: target.
**train_gpu: [0]  workers: 16  batch_size: 16 batch_size_val: 8  base_lr: 0.001  epochs: 10**

-----

[2024-06-14 09:12:47,170 INFO test.py line 248 24] Eval result: mIoU/mAcc/allAcc **0.4915/0.4955/0.9911**.
[2024-06-14 09:12:47,170 INFO test.py line 250 24] Class_0 result: iou/accuracy 0.9829/0.9911, name: background.
[2024-06-14 09:12:47,170 INFO test.py line 250 24] Class_1 result: iou/accuracy 0.0000/0.0000, name: target.
**train_gpu: [0]  workers: 16  batch_size: 16 batch_size_val: 8  base_lr: 0.002  epochs: 50**
epoch [50/50]: mIoU/mAcc/allAcc 0.9124/0.9536/0.9797.
40epoch 之后就不怎么动了  mIoU/mAcc/allAcc 还有个问题 为什么测试的时候数据要比训练的时候好看

-----

../aten/src/ATen/native/cuda/NLLLoss2d.cu:103: nll_loss2d_forward_kernel: block: [5,0,0], thread: [993,0,0] Assertion `t >= 0 && t < n_classes` failed.
maybe transform  yes not randshift  not randcolor is normal   ---> yes randshift no randcolor  is normal  更改randcolor  后正常，因为原来的RandColor把label的也变了，不应该变的，亮度和对比度只对image调整就好了

-----

[2024-06-14 09:38:05,399 INFO test.py line 248 3480] Eval result: mIoU/mAcc/allAcc **0.4899/0.4975/0.9949**.
[2024-06-14 09:38:05,399 INFO test.py line 250 3480] Class_0 result: iou/accuracy 0.9797/0.9949, name: target.
[2024-06-14 09:38:05,399 INFO test.py line 250 3480] Class_1 result: iou/accuracy 0.0000/0.0000, name: background.
a:**train_gpu: [0]  workers: 16  batch_size: 16 batch_size_val: 8  base_lr: 0.002  epochs: 50   rand crop-->crop_type='center'  no Randscale**
后两者让收敛的更快 不知道是不是真的 maybe偶然 epoch38开始不怎么变动  epoch [50/50]: mIoU/mAcc/allAcc 0.9266/0.9595/0.9832
中心裁剪不随机 不放缩大小确实会提升效果
ps 颠倒color name文件的前后顺序 iou为0的类也随之变化  白十字变成黑十字了 确实交换顺序后 又变成白十字了

莫名其妙感觉黑十字效果更好

**这是目前最好的效果，但这是不加rand shift 和color的 不知道为什么**

-----

[2024-06-14 09:57:21,210 INFO test.py line 248 3480] Eval result: mIoU/mAcc/allAcc **0.4907/0.4966/0.9931**.
[2024-06-14 09:57:21,210 INFO test.py line 250 3480] Class_0 result: iou/accuracy 0.9814/0.9931, name: background.
[2024-06-14 09:57:21,210 INFO test.py line 250 3480] Class_1 result: iou/accuracy 0.0000/0.0000, name: target.
b:train_gpu: [0]  workers: 16  batch_size: 16 batch_size_val: 8  base_lr: 0.002  epochs: 50   **+ randshift**
收敛相比于不加 randshift 变慢了 最终效果也不如以前了  是shift太大了 还是写错了 没起到作用  还是还没来得及收敛

-----

random color slowly more  但也许只是因为没收敛  虽然训练时数据变小了 但是测试的结果变好了
不知道是否与shiftmax变小了有关
epoch [50/50]: mIoU/mAcc/allAcc 0.9157/0.9565/0.9804.
2c:024-06-14 15:09:45,999 INFO test.py line 248 24] Eval result: mIoU/mAcc/allAcc 0.4898/0.4971/0.9941.
[2024-06-14 15:09:46,000 INFO test.py line 250 24] Class_0 result: iou/accuracy 0.9796/0.9941, name: background.
[2024-06-14 15:09:46,000 INFO test.py line 250 24] Class_1 result: iou/accuracy 0.0000/0.0000, name: target.
Eval result: mIoU/mAcc/allAcc 0.4902/0.4974/0.9947.  70 epoch 
epoch [70/70]: mIoU/mAcc/allAcc 0.9209/0.9561/0.9818.

-----

only exchange background and target
epoch [50/70]: mIoU/mAcc/allAcc 0.9179/0.9535/0.9811.
epoch [70/70]: mIoU/mAcc/allAcc 0.9248/0.9590/0.9827.
[2024-06-14 15:45:42,582 INFO test.py line 248 4400] Eval result: mIoU/mAcc/allAcc 0.4903/0.4973/0.9946.
[2024-06-14 15:45:42,583 INFO test.py line 250 4400] Class_0 result: iou/accuracy 0.9806/0.9946, name: target.
[2024-06-14 15:45:42,583 INFO test.py line 250 4400] Class_1 result: iou/accuracy 0.0000/0.0000, name: background.

-----

[2024-06-14 17:39:41,170 INFO test.py line 248 24] Eval result: mIoU/mAcc/allAcc 0.4901/0.4962/0.9925.
[2024-06-14 17:39:41,170 INFO test.py line 250 24] Class_0 result: iou/accuracy 0.9802/0.9925, name: background.
[2024-06-14 17:39:41,170 INFO test.py line 250 24] Class_1 result: iou/accuracy 0.0000/0.0000, name: target.
+randcolor 更改顺序 padding=mean->0  越来越差了

-----

训练101 效果稍有起色 但不多 收敛更慢了

[50/50]: mIoU/mAcc/allAcc 0.9169/0.9543/0.9808.  shift=2   看着是未收敛
[2024-06-15 08:56:11,021 INFO test.py line 248 25] Eval result: mIoU/mAcc/allAcc 0.4900/0.4968/0.9936.
[2024-06-15 08:56:11,021 INFO test.py line 250 25] Class_0 result: iou/accuracy 0.9801/0.9936, name: background.
[2024-06-15 08:56:11,021 INFO test.py line 250 25] Class_1 result: iou/accuracy 0.0000/0.0000, name: target.

epoch [50/50]: mIoU/mAcc/allAcc 0.9221/0.9565/0.9821.
[2024-06-15 09:09:39,138 INFO test.py line 248 3480] Eval result: mIoU/mAcc/allAcc 0.4907/0.4967/0.9935.
[2024-06-15 09:09:39,138 INFO test.py line 250 3480] Class_0 result: iou/accuracy 0.9814/0.9935, name: background.
[2024-06-15 09:09:39,138 INFO test.py line 250 3480] Class_1 result: iou/accuracy 0.0000/0.0000, name: target.
base_lr: 0.004



50+20epoch 效果更差

[2024-06-15 09:46:47,279 INFO test.py line 248 103428] Eval result: mIoU/mAcc/allAcc 0.4907/0.4963/0.9926.
[2024-06-15 09:46:47,279 INFO test.py line 250 103428] Class_0 result: iou/accuracy 0.9814/0.9926, name: background.
[2024-06-15 09:46:47,279 INFO test.py line 250 103428] Class_1 result: iou/accuracy 0.0000/0.0000, name: target.
epoch [50/50]: mIoU/mAcc/allAcc 0.9192/0.9567/0.9813.  101 epoch 50 未收敛  70epoch 50也收敛了  val result: mIoU/mAcc/allAcc 0.4909/0.4964/0.9928

-----
pspnet
old:label/255 dont set ignore-label
new: png-255 to 1 set ignore_label=255 
epoch [50/50]: mIoU/mAcc/allAcc 0.9155/0.9540/0.9803
[2024-06-16 16:02:50,422 INFO test.py line 248 3480] Eval result: mIoU/mAcc/allAcc 0.9051/0.9361/0.9821.
[2024-06-16 16:02:50,422 INFO test.py line 250 3480] Class_0 result: iou/accuracy 0.9804/0.9935, name: background.
[2024-06-16 16:02:50,422 INFO test.py line 250 3480] Class_1 result: iou/accuracy 0.8299/0.8787, name: target.

lr 0.002->0.01

[2024-06-17 13:15:47,670 INFO test.py line 254 3480] Eval result: mIoU/mAcc/allAcc 0.9203/0.9533/0.9849.
[2024-06-17 13:15:47,670 INFO test.py line 256 3480] Class_0 result: iou/accuracy 0.9834/0.9927, name: background.
[2024-06-17 13:15:47,670 INFO test.py line 256 3480] Class_1 result: iou/accuracy 0.8573/0.9139, name: target.
epoch [50/50]: mIoU/mAcc/allAcc 0.9275/0.9606/0.9832 
[2024-06-17 13:36:41,268 INFO test.py line 254 2560] Eval result: mIoU/mAcc/allAcc 0.9271/0.9574/0.9862.
[2024-06-17 13:36:41,268 INFO test.py line 256 2560] Class_0 result: iou/accuracy 0.9848/0.9934, name: background.
[2024-06-17 13:36:41,268 INFO test.py line 256 2560] Class_1 result: iou/accuracy 0.8693/0.9213, name: target.
Train result at epoch [80/80]: mIoU/mAcc/allAcc 0.9375/0.9689/0.9856.

[2024-06-17 13:44:53,689 INFO test.py line 254 2100] Eval result: mIoU/mAcc/allAcc 0.9296/0.9576/0.9868.
[2024-06-17 13:44:53,689 INFO test.py line 256 2100] Class_0 result: iou/accuracy 0.9854/0.9940, name: background.
[2024-06-17 13:44:53,689 INFO test.py line 256 2100] Class_1 result: iou/accuracy 0.8738/0.9213, name: target.
epoch [100/100]: mIoU/mAcc/allAcc 0.9397/0.9680/0.9862.
80-100 没什么增长 趋于平稳 不是不涨 是增速变慢了


[2024-06-17 14:04:14,605 INFO test.py line 254 3480] Eval result: mIoU/mAcc/allAcc 0.9381/0.9644/0.9884.
[2024-06-17 14:04:14,605 INFO test.py line 256 3480] Class_0 result: iou/accuracy 0.9872/0.9944, name: background.
[2024-06-17 14:04:14,605 INFO test.py line 256 3480] Class_1 result: iou/accuracy 0.8890/0.9343, name: target.
epoch [150/150]: mIoU/mAcc/allAcc 0.9542/0.9752/0.9896.

[2024-06-17 15:28:29,945 INFO test.py line 254 5780] Eval result: mIoU/mAcc/allAcc 0.9506/0.9729/0.9908.
[2024-06-17 15:28:29,945 INFO test.py line 256 5780] Class_0 result: iou/accuracy 0.9898/0.9953, name: background.
[2024-06-17 15:28:29,945 INFO test.py line 256 5780] Class_1 result: iou/accuracy 0.9114/0.9505, name: target.
epoch [250/250]: mIoU/mAcc/allAcc 0.9685/0.9839/0.9929.
-----

-----
还是unet试一试吧 batch_size =8
mIoU: 95.24; mPA: 97.92; Accuracy: 99.03        50+50

mIoU: 95.91; mPA: 98.23; Accuracy: 99.17        50+50+50  
best-pth
===>background: Iou-99.04; Recall (equal to the PA)-99.4; Precision-99.64
===>target:     Iou-92.48; Recall (equal to the PA)-97.04; Precision-95.17
===> mIoU: 95.76; mPA: 98.22; Accuracy: 99.14
last-pth  selected
===>background: Iou-99.08; Recall (equal to the PA)-99.44; Precision-99.64
===>target:     Iou-92.76; Recall (equal to the PA)-97.02; Precision-95.47
===> mIoU: 95.92; mPA: 98.23; Accuracy: 99.17

mIoU: 95.97; mPA: 98.06; Accuracy: 99.19   50+50+50+50      last
===>background: Iou-99.09; Recall (equal to the PA)-99.5; Precision-99.59
===>target:     Iou-92.85; Recall (equal to the PA)-96.61; Precision-95.98
===> mIoU: 95.97; mPA: 98.06; Accuracy: 99.19
50+50+50+50+50 no progress


-----

## 添加unet
sc create --name seg-u  --image "harbor.smoa.cc/public/xsemseg:v1.4" --gpu 1 --cmd "source /dataset/vsitongwu/.bashrc && cd /dataset/vsitongwu/LJC/TargetSeg/semseg && conda activate animate && sh tool/train.sh target unet50" --arch ampere --debug
sc create --name seg-utrain  --image "harbor.smoa.cc/public/xsemseg:v1.4" --gpu 1 --cmd "source /dataset/vsitongwu/.bashrc && cd /dataset/vsitongwu/LJC/TargetSeg/semseg && conda activate animate && sh tool/train.sh target unet50" --arch ampere 

sc create --name seg-u  --image "harbor.smoa.cc/public/xsemseg:v1.4" --gpu 1 --cmd "source /dataset/vsitongwu/.bashrc && cd /dataset/vsitongwu/LJC/TargetSeg/semseg && conda activate animate && sh tool/mytest.sh target unet50" --arch ampere --debug

TypeError: UNet.forward() takes 2 positional arguments but 3 were given
### 报错解决

### 训练效果
epoch [50/50]: mIoU/mAcc/allAcc 0.8022/0.8830/0.9548
[2024-06-17 11:51:46,024 INFO test.py line 254 24] Eval result: mIoU/mAcc/allAcc 0.8431/0.9261/0.9666.
[2024-06-17 11:51:46,024 INFO test.py line 256 24] Class_0 result: iou/accuracy 0.9634/0.9767, name: background.
[2024-06-17 11:51:46,024 INFO test.py line 256 24] Class_1 result: iou/accuracy 0.7228/0.8756, name: target.

epoch [50/50]: mIoU/mAcc/allAcc 0.7992/0.8817/0.9538.
[2024-06-17 12:13:20,263 INFO test.py line 254 24] Eval result: mIoU/mAcc/allAcc 0.8382/0.9342/0.9645.
[2024-06-17 12:13:20,263 INFO test.py line 256 24] Class_0 result: iou/accuracy 0.9610/0.9720, name: background.
[2024-06-17 12:13:20,263 INFO test.py line 256 24] Class_1 result: iou/accuracy 0.7153/0.8964, name: target.

lr 0.002->0.001
epoch [50/50]: mIoU/mAcc/allAcc 0.7976/0.8888/0.9526. 继续训练20个epoch也没有提高
[2024-06-17 12:19:28,000 INFO test.py line 254 3480] Eval result: mIoU/mAcc/allAcc 0.8289/0.9404/0.9611.
[2024-06-17 12:19:28,000 INFO test.py line 256 3480] Class_0 result: iou/accuracy 0.9572/0.9663, name: background.
[2024-06-17 12:19:28,000 INFO test.py line 256 3480] Class_1 result: iou/accuracy 0.7007/0.9146, name: target.

lr 0.002->0.01
epoch [50/50]: mIoU/mAcc/allAcc 0.9124/0.9504/0.9820.
[2024-06-17 12:50:48,904 INFO test.py line 254 3480] Eval result: mIoU/mAcc/allAcc 0.9248/0.9587/0.9857.
[2024-06-17 12:50:48,904 INFO test.py line 256 3480] Class_0 result: iou/accuracy 0.9842/0.9924, name: background.
[2024-06-17 12:50:48,904 INFO test.py line 256 3480] Class_1 result: iou/accuracy 0.8654/0.9251, name: target.

[80/80]: mIoU/mAcc/allAcc 0.9252/0.9617/0.9846.
[2024-06-17 13:10:12,547 INFO test.py line 254 2560] Eval result: mIoU/mAcc/allAcc 0.9326/0.9592/0.9874.
[2024-06-17 13:10:12,547 INFO test.py line 256 2560] Class_0 result: iou/accuracy 0.9861/0.9943, name: background.
[2024-06-17 13:10:12,547 INFO test.py line 256 2560] Class_1 result: iou/accuracy 0.8791/0.9241, name: target.

epoch [100/100]: mIoU/mAcc/allAcc 0.9295/0.9610/0.9856.
[2024-06-17 13:30:14,353 INFO test.py line 254 2100] Eval result: mIoU/mAcc/allAcc 0.9382/0.9633/0.9885.
[2024-06-17 13:30:14,353 INFO test.py line 256 2100] Class_0 result: iou/accuracy 0.9873/0.9947, name: background.
[2024-06-17 13:30:14,353 INFO test.py line 256 2100] Class_1 result: iou/accuracy 0.8892/0.9318, name: target.

epoch [150/150]: mIoU/mAcc/allAcc 0.9435/0.9700/0.9886.
[2024-06-17 13:38:39,038 INFO test.py line 254 3480] Eval result: mIoU/mAcc/allAcc 0.9530/0.9730/0.9913.
[2024-06-17 13:38:39,038 INFO test.py line 256 3480] Class_0 result: iou/accuracy 0.9904/0.9959, name: background.
[2024-06-17 13:38:39,038 INFO test.py line 256 3480] Class_1 result: iou/accuracy 0.9157/0.9501, name: target.

epoch [200/200]: mIoU/mAcc/allAcc 0.9513/0.9760/0.9902.
[2024-06-17 13:48:43,304 INFO test.py line 254 3480] Eval result: mIoU/mAcc/allAcc 0.9612/0.9767/0.9929.
[2024-06-17 13:48:43,304 INFO test.py line 256 3480] Class_0 result: iou/accuracy 0.9921/0.9969, name: background.
[2024-06-17 13:48:43,304 INFO test.py line 256 3480] Class_1 result: iou/accuracy 0.9304/0.9566, name: target.

epoch [300/300]: mIoU/mAcc/allAcc 0.9650/0.9834/0.9930.
[2024-06-17 14:07:52,429 INFO test.py line 254 5780] Eval result: mIoU/mAcc/allAcc 0.9718/0.9844/0.9948.
[2024-06-17 14:07:52,429 INFO test.py line 256 5780] Class_0 result: iou/accuracy 0.9943/0.9974, name: background.
[2024-06-17 14:07:52,429 INFO test.py line 256 5780] Class_1 result: iou/accuracy 0.9493/0.9714, name: target.

epoch [400/400]: mIoU/mAcc/allAcc 0.9757/0.9888/0.9952.
[2024-06-17 15:24:17,201 INFO test.py line 254 5780] Eval result: mIoU/mAcc/allAcc 0.9770/0.9869/0.9958.
[2024-06-17 15:24:17,201 INFO test.py line 256 5780] Class_0 result: iou/accuracy 0.9954/0.9980, name: background.
[2024-06-17 15:24:17,201 INFO test.py line 256 5780] Class_1 result: iou/accuracy 0.9587/0.9758, name: target.