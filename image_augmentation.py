import argparse
import cv2
import albumentations as A
import os


def parse_args():
    parser = argparse.ArgumentParser(description='Train VGG16 on Garbage classification dataset')
    parser.add_argument('--dataset_dir', type=str, default='dataset/Garbage classification', help='Dataset directory')
    parser.add_argument('--output_dir', type=str, default='augmented', help='Output directory')
    parser.add_argument('--flip', action='store_true', help='Apply horizontal/vertical flips to the images')
    parser.add_argument('--rotate', action='store_true', help='Apply random 90 degree rotations to the images')
    parser.add_argument('--color_shift', action='store_true', help='Apply random color shifts to the images')
    return parser.parse_args()

def main():
    args = parse_args()

    # Declare an augmentation pipeline
    transforms = []
    if args.flip:
        transforms.append((A.HorizontalFlip(p=1), 'horizontal_flip'))
        transforms.append((A.VerticalFlip(p=1), 'vertical_flip'))
    if args.rotate:
        transforms.append((A.Rotate(limit=90, p=1), 'rotate'))
    if args.color_shift:
        transforms.append((A.RGBShift(p=1), 'rgb_shift'))
        transforms.append((A.ChannelShuffle(p=1), 'channel_shuffle'))

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
