import paddlehub as hub
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import numpy as np
import os
from tqdm import tqdm
import cv2
import shutil
from moviepy.editor import *
from paddlehub.dataset.base_cv_dataset import BaseCVDataset


class DemoDataset(BaseCVDataset):
    def __init__(self):
        # 数据集存放位置
        self.dataset_dir = "./train-data"
        super(DemoDataset, self).__init__(
            base_path=self.dataset_dir,
            train_list_file="train_list.txt",
            validate_list_file="validate_list.txt",
            test_list_file="test_list.txt",
            label_list_file="label_list.txt",
        )

# ###---------------------util函数
def checkdir(dir_path):
    '''
    检查是否存在该文件，如果存在就删除了新建，不存在就新建
    :param dir_path:
    :return:
    '''
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
        os.makedirs(dir_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


# 待预测图片
def showForePic(path):
    '''
    传入图片的路径，进行图片的显示
    :param path: 图片的路径
    :return: None
    '''
    img = mpimg.imread(path)
    # 展示待预测图片
    plt.figure(figsize=(10,10))
    plt.imshow(img)
    plt.axis('off')
    plt.show()


def getvideoinfo(video_path):
    '''
    获取视频的帧率和撒小
    :param src_video:
    :return: 帧率，视频大小
    fps: 25.0
    size: (720.0, 576.0)
    '''
    video_capture = cv2.VideoCapture(video_path)
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    size = (video_capture.get(cv2.CAP_PROP_FRAME_WIDTH), video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print("视频的fps: {}\n视频的size: {}".format(fps, size))
    return fps,size


def human_classfication(data):
    '''
    使用前面训练好的图片进行人脸识别分类
    :param data: 要检测的图片的地址
    :return: 人脸的标签（是谁）
    '''
    module = hub.Module(name="resnet_v2_18_imagenet")
    dataset = DemoDataset()

    # 模型构建
    data_reader = hub.reader.ImageClassificationReader(
        image_width=module.get_expected_image_width(),
        image_height=module.get_expected_image_height(),
        images_mean=module.get_pretrained_images_mean(),
        images_std=module.get_pretrained_images_std(),
        dataset=dataset)

    config = hub.RunConfig(
        use_cuda=False,  # 是否使用GPU训练，默认为False；
        num_epoch=4,  # Fine-tune的轮数；
        checkpoint_dir="cv_finetune",  # 模型checkpoint保存路径, 若用户没有指定，程序会自动生成；
        batch_size=10,  # 训练的批大小，如果使用GPU，请根据实际情况调整batch_size；
        eval_interval=10,  # 模型评估的间隔，默认每100个step评估一次验证集；
        strategy=hub.finetune.strategy.DefaultFinetuneStrategy())  # Fine-tune优化策略；
    # 组建FinetuneTask
    input_dict, output_dict, program = module.context(trainable=True)
    img = input_dict["image"]
    feature_map = output_dict["feature_map"]
    feed_list = [img.name]

    task = hub.ImageClassifierTask(
        data_reader=data_reader,
        feed_list=feed_list,
        feature=feature_map,
        num_classes=dataset.num_labels,
        config=config)

    task.load_checkpoint()

    # ##--------------开始预测

    label_map = dataset.label_dict()
    index = 0
    run_states = task.predict(data=data)
    results = [run_state.run_results for run_state in run_states]
    for batch_result in results:
        batch_result = np.argmax(batch_result, axis=2)[0]
        for result in batch_result:
            return result


# ###---------------------util函数


# step1 从视频中提取图片
def extract_images(src_video, dst_dir):
    '''
    src_video:为目标的视频文件地址
    dst_dir:为视频图片的保存路径
    '''
    video = cv2.VideoCapture(src_video)
    count = 0
    while True:
        flag, frame = video.read()
        if not flag:
            break
        cv2.imwrite(os.path.join(dst_dir, str(count) + '.png'), frame)
        count = count + 1
    print('extracted {} frames in total.'.format(count))


# ##step2 对图片进行王者荣耀贴图
def face_chartlet(img_path,save_path):
    module = hub.Module(name="pyramidbox_lite_mobile")
    offset_length = 30
    img = cv2.imread(img_path)

    # prep mask
    masks = [
        cv2.imread('./face/dc.png', -1),
        cv2.imread('./face/wzj.png', -1),
        cv2.imread('./face/xs.png', -1),
        cv2.imread('./face/yyh.png', -1)
    ]

    # set input dict 因为这个模型value为待检测的图片，numpy.array类型，shape为[H, W, C]，BGR格式。所有使用cv2读取
    input_dict = {"data": [img]}
    results = module.face_detection(data=input_dict)

    img2 = img.copy()
    # 因为计划一次只传入一张图片，所有只需要取results的第一个元素就行了
    result_list = results[0]['data']
    if len(result_list) < 1:
        print('图中没有人脸，不做处理')

    elif len(result_list) == 1:
        print('图中只有一张人脸，先进行人脸识别，识别出对应的人脸后再进行贴图')
        index_ = human_classfication([img_path])
        for pos_result in result_list:
            # 获取人脸的位置
            left_pos, right_pos, top_pos, bottom_pos, _ = pos_result.values()
            # print(left_pos, right_pos, top_pos, bottom_pos)
            left_pos = int(left_pos)
            right_pos = int(right_pos)
            top_pos = int(top_pos)
            bottom_pos = int(bottom_pos)
            mask = cv2.resize(masks[index_],
                              (int(right_pos - left_pos + offset_length), int(bottom_pos - top_pos + offset_length)))

            # 取出mask非0的值
            index = mask[:, :, 3] != 0
            index = np.repeat(index[:, :, np.newaxis], axis=2, repeats=3)
            try:
                img2[int(top_pos - offset_length / 2):int(bottom_pos + offset_length / 2), int(left_pos - offset_length / 2):int(right_pos + offset_length / 2), :][index] = mask[:, :, :3][index]
            except Exception as e:
                print(e)

    elif len(result_list) > 1:
        print('检测出了多张人脸，采用随机贴图的方式进行贴图')
        count = 0
        for pos_result in result_list:
            # 获取人脸的位置
            left_pos, right_pos, top_pos, bottom_pos, _ = pos_result.values()
            # print(left_pos, right_pos, top_pos, bottom_pos)
            left_pos = int(left_pos)
            right_pos = int(right_pos)
            top_pos = int(top_pos)
            bottom_pos = int(bottom_pos)
            mask = cv2.resize(masks[count],
                              (int(right_pos - left_pos + offset_length), int(bottom_pos - top_pos + offset_length)))
            # mask = cv2.resize(masks[random.randint(0,3)], (int(right_pos - left_pos), int(bottom_pos - top_pos)))

            # 取出mask非0的值
            index = mask[:, :, 3] != 0
            index = np.repeat(index[:, :, np.newaxis], axis=2, repeats=3)
            try:
                img2[int(top_pos - offset_length / 2):int(bottom_pos + offset_length / 2), int(left_pos - offset_length / 2):int(right_pos + offset_length / 2), :][index] = mask[:, :, :3][index]
            except Exception as e:
                print(e)

            count += 1

    # 保存图片
    cv2.imwrite(os.path.join(save_path, os.path.basename(img_path)), img2)


###step3
# image2video
#将进过face贴图处理的图片从新合成为视频
def img2video(dst_video_path,pic_path,size,frame):
    '''
    dst_video_path:合成视频的保存路径(包含文件名）
    pic_path:合成的所有图片的路径
    size：图片的大小，即是视频的大小
    frame:帧率

    VideoWriter_fourcc为视频编解码器
    fourcc意为四字符代码（Four-Character Codes），顾名思义，该编码由四个字符组成,下面是VideoWriter_fourcc对象一些常用的参数,注意：字符顺序不能弄混
    cv2.VideoWriter_fourcc('I', '4', '2', '0'),该参数是YUV编码类型，文件名后缀为.avi
    cv2.VideoWriter_fourcc('P', 'I', 'M', 'I'),该参数是MPEG-1编码类型，文件名后缀为.avi
    cv2.VideoWriter_fourcc('X', 'V', 'I', 'D'),该参数是MPEG-4编码类型，文件名后缀为.avi
    cv2.VideoWriter_fourcc('T', 'H', 'E', 'O'),该参数是Ogg Vorbis,文件名后缀为.ogv
    cv2.VideoWriter_fourcc('F', 'L', 'V', '1'),该参数是Flash视频，文件名后缀为.flv
    cv2.VideoWriter_fourcc('m', 'p', '4', 'v')    文件名后缀为.mp4
    '''

    dst_video = cv2.VideoWriter(dst_video_path, cv2.VideoWriter_fourcc(*'mp4v'), frame, size, True)
    for index in range(len(os.listdir(pic_path))):
        frame = cv2.imread(os.path.join(pic_path,'{}.png'.format(index)))
        dst_video.write(frame)
    dst_video.release()

###step4 给视频添加声音
def add_audio(s_video_path,d_video_path):
    '''
    给视频加声音
    :param s_video_path: 原视频地址-含有声音的
    :param d_video_path: 目的视频地址-需要加声音的
    :return:
    '''
    video_s = VideoFileClip(s_video_path)
    video_d = VideoFileClip(d_video_path)
    audio_o = video_s.audio
    video_dd = video_d.set_audio(audio_o)
    video_dd.write_videofile(d_video_path[0:d_video_path.rfind('/')+1]+'hong_audio.mp4')


if __name__ == '__main__':
    src_video = '.video/hong4-sing.mov'
    dst_img_dir = './video-imgs'  #每一帧图片的保存路径

    # # step1 extract images of the videos
    checkdir(dst_img_dir)
    extract_images(src_video,dst_img_dir)

    # step2 give the face chartlet of each pic
    save_video_imgs = './video-imgs-face' #贴图图片的保存路径
    checkdir(save_video_imgs)
    for i in tqdm(os.listdir(dst_img_dir)):
        face_chartlet(os.path.join(dst_img_dir,i),save_video_imgs)

    # step3 image2video
    dst_video_path = './video-result'  #最终合成视频的保存路径
    checkdir(dst_video_path)
    video_name = 'hong-sing.mp4'
    dst_video_name = os.path.join(dst_video_path,video_name)
    fps,size = getvideoinfo(src_video)
    size = (int(size[0]),int(size[1]))
    img2video(dst_video_name, save_video_imgs, size, int(fps))

    # step4 add the audio
    add_audio(src_video, dst_video_name)