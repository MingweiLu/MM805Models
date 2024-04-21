import argparse
import cv2
import albumentations as A
import os


def parse_args():
    parser = argparse.ArgumentParser(description='Train VGG16 on Garbage classification dataset')
    parser.add_argument('--dataset_dir', type=str, default='dataset/Garbage classification', help='Dataset directory')
    parser.add_argument('--output_dir', type=str, default='augmented', help='Output directory')
    return parser.parse_args()

def main():
    args = parse_args()

    # Declare an augmentation pipeline
    transforms = [
        (A.RandomCrop(width=256, height=256), 'random_crop'),
        (A.HorizontalFlip(p=1), 'horizontal_flip'),
        (A.VerticalFlip(p=1), 'vertical_flip'),
        (A.RandomRotate90(p=1), 'random_rotate_90'),
        (A.RGBShift(p=1), 'rgb_shift'),
        (A.ChannelShuffle(p=1), 'channel_shuffle'),
    ]

    os.makedirs(args.output_dir)
    for class_name in os.listdir(args.dataset_dir):
        class_dir = os.path.join(args.dataset_dir, class_name)
        output_class_dir = os.path.join(args.output_dir, class_name)
        os.makedirs(output_class_dir)
        for image_name in os.listdir(class_dir):
            image_path = os.path.join(class_dir, image_name)
            image = cv2.imread(image_path)
            for transform, transform_name in transforms:
                augmented = transform(image=image)
                augmented_image = augmented['image']
                output_image_path = os.path.join(output_class_dir, f'{image_name}_{transform_name}.jpg')
                cv2.imwrite(output_image_path, augmented_image)


if __name__ == '__main__':
    main()
