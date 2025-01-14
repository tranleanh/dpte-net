import os
import cv2
import numpy as np

from PIL import Image
from random import randint
from tqdm import tqdm

from .dcp import estimate_transmission
from .networks import img_size
from .networks import EDN_GTM

# img_size = 512

RESHAPE = (img_size,img_size)

def is_an_image_file(filename):
    IMAGE_EXTENSIONS = ['.png', '.jpg', '.jpeg', 'bmp']
    for ext in IMAGE_EXTENSIONS:
        if ext in filename:
            return True
    return False


def list_image_files(directory):
    files = sorted(os.listdir(directory))
    return [os.path.join(directory, f) for f in files if is_an_image_file(f)]


def load_image(path):
    img = Image.open(path)
    return img


def preprocess_image(cv_img):
    cv_img = cv_img.resize(RESHAPE)
    img = np.array(cv_img)
    img = (img - 127.5) / 127.5
    return img


def deprocess_image(img):
    img = img * 127.5 + 127.5
    return img.astype('uint8')


def save_image(np_arr, path):
    img = np_arr * 127.5 + 127.5
    im = Image.fromarray(img)
    im.save(path)


def load_images(path, n_images):
    if n_images < 0:
        n_images = float("inf")
    A_paths, B_paths = os.path.join(path, 'A'), os.path.join(path, 'B')
    all_A_paths, all_B_paths = list_image_files(A_paths), list_image_files(B_paths)
    images_A, images_B = [], []
    images_A_paths, images_B_paths = [], []
    cnt=0
    for path_A, path_B in zip(all_A_paths, all_B_paths):
        img_A, img_B = load_image(path_A), load_image(path_B)
        images_A.append(preprocess_image(img_A))
        images_B.append(preprocess_image(img_B))
        images_A_paths.append(path_A)
        images_B_paths.append(path_B)
        cnt+=1
        print(cnt, n_images)
        if len(images_A) > n_images - 1: break

    return {
        'A': np.array(images_A),
        'A_paths': np.array(images_A_paths),
        'B': np.array(images_B),
        'B_paths': np.array(images_B_paths)
    }


def preprocess_cv2_image(cv_img):
    cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    cv_img = cv2.resize(cv_img, RESHAPE)
    img = np.array(cv_img)
    img = (img - 127.5) / 127.5
    return img


def preprocess_depth_img(cv_img):
    cv_img = cv2.resize(cv_img, RESHAPE)
    img = np.array(cv_img)
    img = np.reshape(img, (RESHAPE[0], RESHAPE[1], 1))
    img = 2*(img - 0.5)
    return img


def preprocess_guide(cv_img):
    cv_img = cv2.resize(cv_img, RESHAPE)
    img = np.array(cv_img)
    return img


def gen_hazy_clean_hint_soft_trans(img_A, img_B, teacher_model):

    t = estimate_transmission(img_A)

    t_flip = cv2.flip(t, 1)

    img_A = cv2.cvtColor(img_A, cv2.COLOR_BGR2RGB)
    img_B = cv2.cvtColor(img_B, cv2.COLOR_BGR2RGB)

    img_A_flip = cv2.flip(img_A, 1)
    img_B_flip = cv2.flip(img_B, 1)

    h, w, _ = img_A.shape
    min_wh = np.amin([h, w])
    crop_sizes = [int(min_wh*0.4), int(min_wh*0.5), int(min_wh*0.6), int(min_wh*0.7), int(min_wh*0.8)]


    images_A = []
    images_B = []
    hints = []
    soft_labels = []
    guides = []


    for crop_size in crop_sizes:

        x1, y1 = randint(1, w-crop_size-1), randint(1, h-crop_size-1)

        # Original of Crop

        # Hazy
        cropA = img_A[y1:y1+crop_size,x1:x1+int(crop_size/h*w)]
        cropA = cv2.resize(cropA, (RESHAPE))
        cropA = np.array(cropA)
        cropA = (cropA - 127.5) / 127.5

        crop_t = t[y1:y1+crop_size,x1:x1+int(crop_size/h*w)]
        guide = 1 - preprocess_guide(crop_t)
        guide = np.reshape(guide, (RESHAPE[0], RESHAPE[1], 1))
        crop_t = preprocess_depth_img(crop_t)

        cropA = np.concatenate((cropA, crop_t), axis=2)


        # Hint
        teacher_input = np.reshape(cropA, (1,img_size,img_size,4))
        soft_label, hint_fm = teacher_model.predict(x=teacher_input)


        # Clean
        cropB = img_B[y1:y1+crop_size,x1:x1+int(crop_size/h*w)]
        cropB = cv2.resize(cropB, (RESHAPE))
        cropB = np.array(cropB)
        cropB = (cropB - 127.5) / 127.5

        images_A.append(cropA)
        images_B.append(cropB)
        hints.append(hint_fm[0])
        soft_labels.append(soft_label[0])
        guides.append(guide)


        # Horizontal Flip of Crop

        # Hazy
        cropA = img_A_flip[y1:y1+crop_size,x1:x1+int(crop_size/h*w)]
        cropA = cv2.resize(cropA, (RESHAPE))
        cropA = np.array(cropA)
        cropA = (cropA - 127.5) / 127.5

        crop_t = t_flip[y1:y1+crop_size,x1:x1+int(crop_size/h*w)]
        guide = 1 - preprocess_guide(crop_t)
        guide = np.reshape(guide, (RESHAPE[0], RESHAPE[1], 1))
        crop_t = preprocess_depth_img(crop_t)

        cropA = np.concatenate((cropA, crop_t), axis=2)


        # Hint
        teacher_input = np.reshape(cropA, (1,img_size,img_size,4))
        soft_label, hint_fm = teacher_model.predict(x=teacher_input)


        # Clean
        cropB = img_B_flip[y1:y1+crop_size,x1:x1+int(crop_size/h*w)]
        cropB = cv2.resize(cropB, (RESHAPE))
        cropB = np.array(cropB)
        cropB = (cropB - 127.5) / 127.5

        images_A.append(cropA)
        images_B.append(cropB)
        hints.append(hint_fm[0])
        soft_labels.append(soft_label[0])
        guides.append(guide)


    # Original

    # Hazy
    img_A = cv2.resize(img_A, (RESHAPE))
    img_A = np.array(img_A)
    img_A = (img_A - 127.5) / 127.5
    guide = 1 - preprocess_guide(t)
    guide = np.reshape(guide, (RESHAPE[0], RESHAPE[1], 1))
    t = preprocess_depth_img(t)
    img_A = np.concatenate((img_A, t), axis=2)


    # Hint
    teacher_input = np.reshape(img_A, (1,img_size,img_size,4))
    soft_label, hint_fm = teacher_model.predict(x=teacher_input)


    # Clean
    img_B = cv2.resize(img_B, (RESHAPE))
    img_B = np.array(img_B)
    img_B = (img_B - 127.5) / 127.5

    images_A.append(img_A)
    images_B.append(img_B)
    hints.append(hint_fm[0])
    soft_labels.append(soft_label[0])
    guides.append(guide)


    # Horizontal Flip

    # Hazy
    img_A = cv2.resize(img_A_flip, (RESHAPE))
    img_A = np.array(img_A)
    img_A = (img_A - 127.5) / 127.5
    guide = 1 - preprocess_guide(t_flip)
    guide = np.reshape(guide, (RESHAPE[0], RESHAPE[1], 1))
    t = preprocess_depth_img(t_flip)
    img_A = np.concatenate((img_A, t), axis=2)


    # Hint
    teacher_input = np.reshape(img_A, (1,img_size,img_size,4))
    soft_label, hint_fm = teacher_model.predict(x=teacher_input)


    # Clean
    img_B = cv2.resize(img_B_flip, (RESHAPE))
    img_B = np.array(img_B)
    img_B = (img_B - 127.5) / 127.5

    images_A.append(img_A)
    images_B.append(img_B)
    hints.append(hint_fm[0])
    soft_labels.append(soft_label[0])
    guides.append(guide)

    return images_A, images_B, hints, soft_labels, guides


def load_hazy_clean_hint_soft_trans(path, n_images):

    if n_images < 0:
        n_images = float("inf")
    A_paths, B_paths = os.path.join(path, 'A'), os.path.join(path, 'B')
    all_A_paths, all_B_paths = list_image_files(A_paths), list_image_files(B_paths)
    images_A_paths, images_B_paths = [], []

    cnt=0

    print('\n Loading teacher network:')
    teacher_model = EDN_GTM()
    weight_path = 'path/to/teacher_weights.h5'
    teacher_model.load_weights(weight_path)
    print('-> Loading teacher network: Done.')


    print('\n Loading & augmenting training data:')

    images_A = []
    images_B = []
    images_C = []
    images_D = []
    images_E = []


    for path_A, path_B in tqdm(zip(all_A_paths, all_B_paths), total=len(all_A_paths)):

        img_A, img_B = cv2.imread(path_A), cv2.imread(path_B)

        img_A = cv2.resize(img_A, RESHAPE)
        img_B = cv2.resize(img_B, RESHAPE)

        processed_imgs_A, processed_imgs_B, generated_hints, soft_labels, guides = gen_hazy_clean_hint_soft_trans(img_A, img_B, teacher_model)

        for imgA in processed_imgs_A: images_A.append(imgA)
        for imgB in processed_imgs_B: images_B.append(imgB)
        for imgC in generated_hints: images_C.append(imgC)
        for imgD in soft_labels: images_D.append(imgD)
        for imgE in guides: images_E.append(imgE)

        images_A_paths.append(path_A)
        images_B_paths.append(path_B)

        cnt+=1
        if cnt >= n_images: break
        
    return {
        'A': np.array(images_A),
        'B': np.array(images_B),
        'C': np.array(images_C),
        'D': np.array(images_D),
        'E': np.array(images_E)
    }