# -!- coding: utf-8 -!-
import tifffile
import numpy as np

def text_save_sgems(filename, data, d, h, w):
    # filename为写入CSV文件的路径，data为要写入数据列表.
    file = open(filename, 'a')
    file.write(str(d) + ' ' + str(h) + ' ' + str(w) + '\n')
    file.write('1' + '\n')
    file.write('data' + '\n')
    # for i in range(len(data)):
    #     s = str(data[i]).replace('[', '').replace(']', '')  # 去除[],这两行按数据不同，可以选择
    #     s = s.replace("'", '').replace(',', '') + '\n'  # 去除单引号，逗号，每行末尾追加换行符
    #     file.write(s)
    # file.close()
    # print("保存文件成功")
    for i in range(len(data)):
        if data[i] >= 0.5:
            file.write('0\n')
        else:
            file.write('1\n')
    file.close()
    print("保存文件成功")

sample = 1501
# while 20<sample<31:

if __name__ == '__main__':
    img_sys_path='TrainedModels/shale40/2022_05_03_12_59_03_generation_train_depth_1_lr_scale_0.2_act_lrelu_0.05/5'
    im_in = tifffile.imread('%s/fake_sample_fake_%d.tiff' % (img_sys_path, sample))
    #im_in = tifffile.imread('%s/real_scale.tiff' % (img_sys_path))
    #data = (im_in/255).astype(np.int32).reshape(80 * 80 * 80, 3)
    #data = im_in.astype(np.int32).reshape(80 * 80 * 80, 1)
    data = im_in.astype(np.float).reshape(80 * 80 * 80, 1)
    txt_path = '%s/shale_%d.txt' % (img_sys_path, sample)
    #txt_path = '%s/shale.txt' % (img_sys_path)
    text_save_sgems(txt_path, data, 80, 80, 80)

    # img_sys_path = '../SRATSIGAN/Images/Generation'
    # im_in = tifffile.imread('%s/berea_80.tif' % (img_sys_path))
    # #data = im_in.astype(np.float).reshape(80 * 80 * 80, 1)
    # data = (im_in / 255).astype(np.int32).reshape(80 * 80 * 80, 3)
    # txt_path = '%s/berea_80.txt' % (img_sys_path)
    # text_save_sgems(txt_path, data, 80, 80, 80)