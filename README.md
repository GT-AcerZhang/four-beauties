# 楔子
西施沉鱼自苎萝 当为复越首功臣

昭君落雁出边塞 一平烽火即五旬

貂蝉月下邀对酌 离间巧计除奸恶

玉环香消马嵬坡 舍身为国安大唐----《王者四大美女--红昭愿》


## 目的
**将b站很火的4小戏骨的红昭愿视频中相信很多人都看过，
里面的角色背景是我国古代四大美女。本人又是一个王者荣耀er，
所有就想到了将王者荣耀中四大美女的形象人脸替换到B站视频上面的创意。**
## 项目地址
[aistudio地址](https://aistudio.baidu.com/aistudio/projectdetail/454586)

[github地址]()

## 使用到的模型：
模型1[pyramidbox_lite_mobile](https://www.paddlepaddle.org.cn/hubdetail?name=pyramidbox_lite_mobile&en_category=ObjectDetection)

模型2[resnet_v2_18_imagenet](https://www.paddlepaddle.org.cn/hubdetail?name=resnet_v2_18_imagenet&en_category=ImageClassification)
## 实现思路
        	1.先将视频按帧提取保存为图片的集合
            2.然后使用paddlehub提供的pyramidbox_lite_mobile模型进行人脸位置检测，使用自己finetune的resnet_v2_18_imagenet模型对每一帧图片进行人脸识别。可能出现图片中没有人、图片中只有一个人、图片中有超过一个人三种情况。对于图中没有人，则不做任何操作；如果图中只有一个人，那么可以使用模型2精确分类该人物是谁，然后换上对应的贴图；如果图中使用模型1识别处理超过1人，则采用随机算法，随机给人脸换上贴图。
        	3.将经过人脸分类+识别+贴图的图片集合使用cv2生成视频
            4.步骤3中生成的视频是没有bgm的，采用VideoFileClip给视频加上bgm

## How to Run
* 将自己的视频放到video文件夹中
* 更改main函数中258行的目标视频文件的名字
* 然后允许main函数即可。
* 最终会在当前目录的video-result目录生成两个视频文件，一个带有bgm一个不带bgm
