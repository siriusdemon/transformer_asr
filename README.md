# 文件说明
 - transformerapp ： 做成接口的代码
 - transformer_train : 训练模型的代码

# api
+ 生成的[url:](http://192.168.9.202:27705/SpeechRecognition/),端口映射：audio:file,mp3,wav格式，
+ ![NHCC2F.png](https://s1.ax1x.com/2020/07/01/NHCC2F.png)

# 训练过程
+ 0、训练在transformer_djl中，开启环境：`source activate /envs/transformer`
+ 1、通过createcsv.py文件生成 三个 csv文件
+ 2、通过build_featurecab.py 生成.t文件
> 产生的.t 文件相当于字典
> 执行：python build_features_vocab.py 数字是汉字总数
> ![NHCnPK.png](https://s1.ax1x.com/2020/07/01/NHCnPK.png)
+ 3、修改.yaml 配置参数
+ 4、通过main.py训练模型
` CUDA_VISIBLE_DEVICES=0,3 python main.py`
+ 5、通过predict.py预测一句话的效果
`CUDA_VISIBLE_DEVICES=3 python predict.py`
+ 6、通过eval.py预测测试集的准确率
` CUDA_VISIBLE_DEVICES=3 python eval.py`
> cer:0.0852(),avg_cer（每个batch的平均值）:0.0739,_cer（所有的> 字）:0.0743 验证集错误率
+ 