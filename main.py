#@author: Longbin Jin
############################
import os
import cv2
import argparse
import pickle
import base64
from io import BytesIO
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
from torchvision import transforms
from network_inf import builder_inf
from facenet_pytorch import MTCNN


# parse the args
parser = argparse.ArgumentParser(description='Trainer for posenet')
parser.add_argument('--arch', default='iresnet100', type=str,
                    help='backbone architechture')
parser.add_argument('--inf_list', default='/home/lulla/data/class01', type=str,
                    help='the inference list')
parser.add_argument('--class_folder', default='/home/lulla/data/class01', type=str,
                    help='the inference list')
parser.add_argument('--feat_list', type=str,
                    help='The save path for saveing features')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch_size', default=256, type=int, metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                    'batch size of all GPUs on the current node when '
                    'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--embedding_size', default=512, type=int,
                    help='The embedding feature size')
parser.add_argument('--resume', default='/home/lulla/code/magface_epoch_00025.pth', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--cpu-mode', action='store_true', help='Use the CPU.')
args = parser.parse_args()


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

mtcnn = MTCNN(
image_size=112, margin=0, min_face_size=50,
thresholds=[0.6, 0.7, 0.7], factor=0.709, keep_all=True, post_process=False,
device=device)

trans = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0., 0., 0.],
        std=[1., 1., 1.]),
])

model = builder_inf(args)
model.eval()
model.to(device)


def load_base64(base64_string, mode='RGB'):
    """
    이미지 파일을 bytes에서 numpy array로 변환
    """
    image = Image.open(BytesIO(base64.b64decode(base64_string)))
    if mode=='RGB':
        image = image.convert('RGB')
    elif mode == 'BGR':
        image = image.convert('RGB')
        image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
    return np.array(image)


def face_detection(base64_string:bytes, face_location:list=None):
    '''한장의 이미지에서 얼굴 위치와 얼굴 특징 (512차원 벡터) 추출'''

    image = load_base64(base64_string, mode='RGB')

    if not face_location:
        face_location = []
        face_locations, prob = mtcnn.detect(image)
        if prob[0] == None:
            return [], []
        
        for i in range(len(face_locations)):
            if prob[i] > 0.9:
                left, top, right, bottom = map(int, face_locations[i])

                h, w = int(bottom) - int(top), int(right) - int(left)
                if h > w:
                    right += (h-w)//2
                    left -= (h-w)//2
                elif h < w:
                    bottom += (w-h)//2
                    top -= (w-h)//2

                if top < 0:
                    bottom -= top
                    top = 0
                if left < 0:
                    right -= left
                    left = 0
                if bottom > image.shape[0]:
                    top -= bottom - image.shape[0]
                    bottom = image.shape[0]
                if right > image.shape[1]:
                    left -= right - image.shape[1]
                    right = image.shape[1]

                face_location.append([top, right, bottom, left])

    face_encoding = []
    for location in face_location:
        top, right, bottom, left = location
        resized_img = cv2.resize(image[top:bottom, left:right, :], (112, 112), interpolation=cv2.INTER_CUBIC)
        transformed_img = trans(resized_img).unsqueeze(0)
        if torch.cuda.is_available():
            transformed_img = transformed_img.cuda()
        embedding_feat = model(transformed_img)
        face_encoding.append(embedding_feat[0].cpu().detach().numpy())

    return face_location, face_encoding


def save_face_par(base64_string:bytes, img_id:str, child_ids:str, data_folder:str, class_id:str):
    '''
    부모가 처음 앱에 가입할 때 본인 아이 사진을 업로드 하면 
    그 사진들에서 얼굴을 검출하고 512차원 벡터로 추출하여 파일로 저장
    이미지를 byte-string 형식으로 입력
    사진에서 하나의 얼굴이 검출되어야 함
    '''

    # 얼굴 정보를 저장 할 파일
    data_path = os.path.join(data_folder, class_id) + '.pkl'
    if os.path.isfile(data_path):
        with open(data_path,'rb') as f:
            data_dict = pickle.load(f)
    else:
        data_dict = {'known_img_id':[], 'known_child_ids':[],
                     'known_face_locations':[], 'known_face_encodings':[]}

    # 사진에서 얼굴 위치 및 특징 추출
    if img_id in data_dict['known_img_id']:
        return data_dict['known_img_id'], data_dict['known_child_ids'], data_dict['known_face_locations'], data_dict['known_face_encodings']
    
    face_location, face_encoding = face_detection(base64_string)
    
    if len(face_location) != 1:
        return data_dict['known_img_id'], data_dict['known_child_ids'], data_dict['known_face_locations'], data_dict['known_face_encodings']

    data_dict['known_img_id'].append(img_id)
    data_dict['known_child_ids'].append(child_ids)
    data_dict['known_face_locations'].append(face_location[0])
    data_dict['known_face_encodings'].append(face_encoding[0])
        
    with open(data_path,'wb') as f:
        pickle.dump(data_dict, f)

    return data_dict['known_img_id'], data_dict['known_child_ids'], data_dict['known_face_locations'], data_dict['known_face_encodings']


def save_face_tea(base64_string:bytes, img_id:str, face_locations_tag:list, child_ids_tag:list, data_folder:str, class_id:str):
    '''
    선생님이 직접 태깅한 정보를 데이터 베이스에 저장
    한 장의 이미지에서 여러개 얼굴 태깅 가능
    len(face_locations_tag) == len(child_ids_tag)
    '''
    if len(face_locations_tag) != len(child_ids_tag):
        raise Exception('len(face_locations_tag) != len(child_ids_tag)')

    face_locations, face_encodings = face_detection(base64_string, face_location=face_locations_tag)

    # 얼굴 정보를 저장 할 파일
    data_path = os.path.join(data_folder, class_id) + '.pkl'
    if os.path.isfile(data_path):
        with open(data_path,'rb') as f:
            data_dict = pickle.load(f)
    else:
        data_dict = {'known_img_id':[], 'known_child_ids':[],
                     'known_face_locations':[], 'known_face_encodings':[]}
    
    for i, child_id in enumerate(child_ids_tag):
        if (child_id) and (face_locations_tag[i] not in data_dict['known_face_locations']) and (face_locations_tag[i] in face_locations):
            data_dict['known_img_id'].append(img_id)
            data_dict['known_child_ids'].append(child_id)
            data_dict['known_face_locations'].append(face_locations_tag[i])
            data_dict['known_face_encodings'].append(face_encodings[face_locations.index(face_locations_tag[i])])
    
    with open(data_path,'wb') as f:
        pickle.dump(data_dict, f)

    return data_dict['known_img_id'], data_dict['known_child_ids'], data_dict['known_face_locations'], data_dict['known_face_encodings']


def move_child(child_id:str , org_class_id:str , move_class_id:str , data_folder:str):
    # 기존 반 데이터 경로
    before_data_path = os.path.join(data_folder, org_class_id) + '.pkl'
    # 변경된 반 데이터 경로
    after_data_path = os.path.join(data_folder , move_class_id) + '.pkl' 

    # 기존 반 데이터 파일 존재를 확인
    if os.path.isfile(before_data_path):
        with open(before_data_path,'rb') as f:
            before_data_dict = pickle.load(f)
    else:
        raise Exception(f'{before_data_path} is not exist!!')
    
    # 변경된 반 데이터 파일 존재를 확인
    if os.path.isfile(after_data_path):
        with open(after_data_path , 'rb') as f:
            after_data_dict = pickle.load(f)
    else:
        after_data_dict = {'known_base64_string' : []  , 'known_child_ids' : [] ,
                           'known_face_locations' : [] , 'known_face_encodings' : []}

    # 기존 반 데이터에 child_id를 가진 데이터가 없어질 때 까지 반복
    while child_id in before_data_dict['known_child_ids']:
        idx = before_data_dict['known_child_ids'].index( child_id )
        after_data_dict['known_base64_string'].append( before_data_dict['known_base64_string'][idx])
        del before_data_dict['known_base64_string'][idx]
        after_data_dict['known_child_ids'].append( before_data_dict['known_child_ids'][idx])
        del before_data_dict['known_child_ids'][idx]
        after_data_dict['known_face_locations'].append( before_data_dict['known_face_locations'][idx])
        del before_data_dict['known_face_locations'][idx]
        after_data_dict['known_face_encodings'].append( before_data_dict['known_face_encodings'][idx])
        del before_data_dict['known_face_encodings'][idx]

    # 기존 반 데이터 덮어씌우기
    with open(before_data_path , 'wb') as f:
        pickle.dump(before_data_dict , f)

    # 변경된 반 데이터 덮어씌우기 또는 생성
    with open(after_data_path , 'wb') as f:
        pickle.dump(after_data_dict , f)

        
def remove_face(img_id:str, face_locations:list, child_ids:list, data_folder:str, class_id:str):
    '''저장된 데이터에서 얼굴 정보 삭제'''

    if len(face_locations) != len(child_ids):
        raise Exception('len(face_locations_tag) != len(child_ids_tag)')

    data_path = os.path.join(data_folder, class_id) + '.pkl'
    if not os.path.isfile(data_path):
        raise Exception(f'{data_path} not exist!')

    with open(data_path,'rb') as f:
        data_dict = pickle.load(f)

    for i, face_location in enumerate(face_locations):
        if (img_id in data_dict['known_img_id']) and (face_location in data_dict['known_face_locations']) and (child_ids[i] in data_dict['known_child_ids']):
            idx = data_dict['known_face_locations'].index(face_location)
            del data_dict['known_img_id'][idx]
            del data_dict['known_child_ids'][idx]
            del data_dict['known_face_locations'][idx]
            del data_dict['known_face_encodings'][idx]
    
    with open(data_path,'wb') as f:
        pickle.dump(data_dict, f)
        

def cosine_similarity(A, B):
    A, B = np.array(A), np.array(B)
    # Normalize A and B
    A_norm = A / np.linalg.norm(A, axis=1, keepdims=True)
    B_norm = B / np.linalg.norm(B)
    
    # Compute cosine similarity
    similarity = np.dot(A_norm, B_norm)
    return list(similarity)


def face_matching(base64_string:bytes, known_face_encodings:list, tolerance=0.3):
    '''단체 사진에서 검출된 얼굴에 대해 이름 태깅'''

    face_locations, face_encodings = face_detection(base64_string)

    face_names = []
    for face_encoding in face_encodings:
        temp = cosine_similarity(known_face_encodings, face_encoding)
        dis = [tolerance] * len(temp)
        dis[temp.index(max(temp))] = max(temp)
        match =  list(np.array(dis) > tolerance)

        name = None
        if True in match:
            name = known_child_ids[match.index(True)]

        face_names.append(name)
    return face_locations, face_names


def visualize(base64_string:bytes, output_dir:str, img_name:str, face_locations:list, child_ids:list):
    '''단체 사진에서 얼굴 검출 및 매칭 결과를 이미지로 저장'''

    bgr_img = load_base64(base64_string, mode='BGR')
    for (top, right, bottom, left), name in zip(face_locations, child_ids):
        if not name:
            cv2.rectangle(bgr_img, (left, top), (right, bottom), (255, 0, 0), 2)
            continue

        # 얼굴에 박스 그리기
        cv2.rectangle(bgr_img, (left, top), (right, bottom), (255, 0, 0), 2)

        # 박스 아래 이름 달기
        cv2.rectangle(bgr_img, (left, bottom), (right, bottom+20), (255, 0, 0), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(bgr_img, name, (left + 4, bottom + 15), font, 0.4, (255, 255, 255), 1)

    cv2.imwrite(os.path.join(output_dir, img_name), bgr_img)


if __name__ == '__main__':
    class_name = 'class10'
    data_folder = '/home/lulla/data'
    class_folder = os.path.join(data_folder, class_name)
    single_img_dir = os.path.join(class_folder, 'single')
    group_img_dir = os.path.join(class_folder, 'group')
    group_output_dir = os.path.join(class_folder, 'output')
    
    
    os.makedirs(group_output_dir, exist_ok=True)

    single_child_ids = os.listdir(single_img_dir)
    group_img_names = os.listdir(group_img_dir)

    tolerance = 0.2

    print('=== face detection and save data ===')
    for child_id in tqdm(single_child_ids):
        single_child_path = os.path.join(single_img_dir, child_id)
        img_names = os.listdir(single_child_path)

        for i, img_name in enumerate(img_names):
            img_path = os.path.join(single_child_path, img_name)

            with open(img_path, 'rb') as img:
                base64_string = base64.b64encode(img.read())

            known_base64_string, known_child_ids, known_face_locations, known_face_encodings = save_face_par(base64_string, base64_string[-10:], child_id, class_folder, class_name)

    print('=== face matching ===')
    for img_name in tqdm(group_img_names):
        # print(img_name)
        img_path = os.path.join(group_img_dir, img_name)

        with open(img_path, 'rb') as img:
            base64_string = base64.b64encode(img.read())
        face_locations, face_names = face_matching(base64_string, known_face_encodings, tolerance=tolerance)

        if face_names:
            visualize(base64_string, group_output_dir, img_name, face_locations, face_names)