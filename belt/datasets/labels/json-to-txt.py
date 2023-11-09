import json
import os

name2id={'Fastened Belt':0,'Unfastened Belt':1,'Open Window':2,'Closed Window':3,'Person':4}

def convert(img_size, box):
    dw = 1. / (img_size[0])
    dh = 1. / (img_size[1])
    x = (box[0] + box[2]) / 2.0 - 1
    y = (box[1] + box[3]) / 2.0 - 1
    w = box[2] - box[0]
    h = box[3] - box[1]
    x = x * dw
    y = y * dh
    w = w * dw
    h = h * dh
    return (x, y, w, h)

def decode_json(json_folder_path, json_name):
    txt_name = 'D:\\py\\Yolov5\\yolov5\\belt\\datasets\\labels\\train\\' + json_name[0:-5] + '.txt'
    txt_file = open(txt_name, 'w')
    json_path = os.path.join(json_folder_path, json_name)
    data = json.load(open(json_path, 'r', encoding='gb2312',errors='ignore'))
    img_w = data['imageWidth']
    img_h = data['imageHeight']
    
    for i in data['shapes']:
        label_name = i['label']
        if (i['shape_type'] == 'rectangle'):
            x1 = int(i['points'][0][0])
            y1 = int(i['points'][0][1])
            x2 = int(i['points'][1][0])
            y2 = int(i['points'][1][1])
            
            bb = (x1, y1, x2, y2)
            bbox = convert((img_w, img_h), bb)
            txt_file.write(str(name2id[label_name]) + " "+ " ".join([str(a) for a in bbox]) + '\n')
            
if __name__ == '__main__':
    json_folder_path = 'D:\\py\\Yolov5\\yolov5\\belt\\datasets\\labels\\train-json'
    json_names = os.listdir(json_folder_path)
    for json_name in json_names:
        decode_json(json_folder_path, json_name)
            
            