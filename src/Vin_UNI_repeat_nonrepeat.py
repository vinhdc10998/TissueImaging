import cv2
import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(description="Tissue Imaging")
    parser.add_argument("--path_1", type=str, default='./data/Dataset_BUSI_with_GT', help="path of repeat image")
    parser.add_argument("--path_2", type=str, default='./data/Dataset_BUSI_with_GT', help="path of nonRepeat Image")
    parser.add_argument("--output_path", type=str, default='./data/Dataset_BUSI_with_GT', help="path of output Image")

    args = parser.parse_args()
    return args

def get_medium_image(file1, file2, output_path):
    print("[INFO] File 1:", file1)
    print("[INFO] File 2:", file2)

    image_1 = cv2.imread(file1)
    # image_1 = cv2.cvtColor(image_1,cv2.COLOR_BGR2RGB)

    image_2 = cv2.imread(file2)
    # image_2 = cv2.cvtColor(image_2,cv2.COLOR_BGR2RGB)

    image_3 = (image_1 + image_2)
    # print(image_3)
    # image_3 = np.sum(image_1, image_2)

    print(output_path)
    cv2.imwrite(output_path, image_3)
    # # cv2.imshow("2", image_2)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

def make_file_csv(path):
    '''
    0: healthy
    1: disease
    '''
    items = {
        # 'id': [],
        'weight_1': [],
        'weight_2': [],
        'weight_3': [],
        'label': []
    }
    path = 'data/image/crc/healthy'
    weight1 = 'sfs_weight1'
    weight2 = 'sfs_weight2'
    weight3 = 'sfs_weight3'
    path1= os.path.join(path, weight1)
    for i in os.listdir(path1):
        # items['id'].append(i)
        items['weight_1'].append(os.path.join(path1, i))
        items['weight_2'].append(os.path.join(path, weight2, i))
        items['weight_3'].append(os.path.join(path, weight3, i))
        items['label'].append(0)
    print(len(items['weight_1']))
    print(len(items['weight_2']))
    print(len(items['weight_3']))
    print(len(items['label']))

    path = 'data/image/gastric/healthy'
    weight1 = 'sfs_weight1'
    weight2 = 'sfs_weight2'
    weight3 = 'sfs_weight3'
    path1= os.path.join(path, weight1)
    for i in os.listdir(path1):
        # items['id'].append(i)
        items['weight_1'].append(os.path.join(path1, i))
        items['weight_2'].append(os.path.join(path, weight2, i))
        items['weight_3'].append(os.path.join(path, weight3, i))
        items['label'].append(0)
    print(len(items['weight_1']))
    print(len(items['weight_2']))
    print(len(items['weight_3']))
    print(len(items['label']))

    path = 'data/image/gastric/tumor'
    weight1 = 'Sample Frequency Sweep - Weight 1'
    weight2 = 'Sample Frequency Sweep - Weight 2'
    # weight3 = 'sfs_weight3'
    path1= os.path.join(path, weight1)
    for i in os.listdir(path1):
        # items['id'].append(i)
        items['weight_1'].append(os.path.join(path1, i))
        items['weight_2'].append(os.path.join(path, weight2, i))
        items['weight_3'].append(None)
        items['label'].append(1)
    print("STEP tumor gastric")
    print(len(items['weight_1']))
    print(len(items['weight_2']))
    print(len(items['weight_3']))
    print(len(items['label']))

    path = 'data/image/crc/tumor'
    weight3 = 'Sample Frequency Sweep - Weight 3'
    # weight3 = 'sfs_weight3'
    path1= os.path.join(path, weight3)
    for i in os.listdir(path1):
        # items['id'].append(i)
        items['weight_1'].append(None)
        items['weight_2'].append(None)
        items['weight_3'].append(os.path.join(path, weight3, i))
        items['label'].append(1)


    # print(len(items['id']))
    print(len(items['weight_1']))
    print(len(items['weight_2']))
    print(len(items['weight_3']))
    print(len(items['label']))


    df = pd.DataFrame(items)
    df.to_csv('data/train.csv', index=None)

def main():
    args = parse_args()
    repeat_path = args.path_1
    nonRepeat_path = args.path_2
    output_path = args.output_path

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    # for i in os.listdir(repeat_path):
    #     fileName = i
    #     file1 = os.path.join(repeat_path, fileName)
    #     file2 = os.path.join(nonRepeat_path, fileName)
    #     output = os.path.join(output_path, fileName)

    #     get_medium_image(file1, file2, output)
    #     # break
    
    make_file_csv(output_path)



if __name__ == '__main__':
    main()