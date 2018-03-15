import random
import cv2


def network1_augment(sample,vertical_prob,horizontal_prob):
    image, masks = sample['image'], sample['masks']

    if (random.random()<vertical_prob):
        image=cv2.flip(image,1)
        masks=cv2.flip(masks,1)

    if (random.random()<horizontal_prob):
        image=cv2.flip(image,0)
        masks=cv2.flip(masks,0)

    return {'image': image, 'masks': masks}

def network2_augment(sample,vertical_prob,horizontal_prob):
    image, masks, input_net_1 = sample['image'], sample['masks'], sample['input_net_1']

    if (random.random()<vertical_prob):
        image=cv2.flip(image,1)
        masks=cv2.flip(masks,1)
        input_net_1 = cv2.flip(input_net_1,1)

    if (random.random()<horizontal_prob):
        image=cv2.flip(image,0)
        masks=cv2.flip(masks,0)
        input_net_1 = cv2.flip(input_net_1,0)

    return {'image': image, 'masks': masks, 'input_net_1':input_net_1}

def network3_augment(sample,vertical_prob,horizontal_prob):
    image, masks, input_net_1, input_net_2 = sample['image'], sample['masks'], sample['input_net_1'], sample['input_net_2']

    if (random.random()<vertical_prob):
        image=cv2.flip(image,1)
        masks=cv2.flip(masks,1)
        input_net_1 = cv2.flip(input_net_1,1)
        input_net_2 = cv2.flip(input_net_2,1)

    if (random.random()<horizontal_prob):
        image=cv2.flip(image,0)
        masks=cv2.flip(masks,0)
        input_net_1 = cv2.flip(input_net_1,0)
        input_net_2 = cv2.flip(input_net_2,0)

    return {'image': image, 'masks': masks, 'input_net_1':input_net_1, 'input_net_2':input_net_2}