import argparse
import os
import pickle
import time
import numpy as np
from skimage.transform import resize
from skimage.io import imread
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

IMAGE_CLASS_NAMES = ['metal', 'glass', 'paper', 'trash', 'cardboard', 'plastic']
IMAGE_RESIZE = (512 // 2, 384 // 2, 3)

def parse_args():
    parser = argparse.ArgumentParser(description='Train SVM on Garbage classification dataset')
    parser.add_argument('--dataset_dir', type=str, default='dataset/Garbage classification', help='Dataset directory')
    parser.add_argument('--model_save_path', type=str, default='svm.pkl', help='Path to save the model')
    return parser.parse_args()


def load_data(dataset_dir: str, image_class_names: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = []
    y = []
    for i, class_name in enumerate(image_class_names):
        class_dir = os.path.join(dataset_dir, class_name)
        for image_name in os.listdir(class_dir):
            image_path = os.path.join(class_dir, image_name)
            image = imread(image_path)
            image = resize(image, IMAGE_RESIZE)
            x.append(image.flatten())
            y.append(i)
    x = np.array(x)
    y = np.array(y)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)
    return x_train, x_test, y_train, y_test


def train_svm(x_train: np.ndarray, y_train: np.ndarray) -> svm.SVC:
    """
    ref: https://www.geeksforgeeks.org/image-classification-using-support-vector-machine-svm-in-python/
    """
    param_grid = {'C': [0.1, 1, 10], 'gamma': [0.0005, 0.001, 0.01], 'kernel': ['linear']}
    model = GridSearchCV(svm.SVC(probability=True), param_grid)
    model.fit(x_train, y_train)
    return model


def evaluate_svm(model: svm.SVC, x_test: np.ndarray, y_test: np.ndarray) -> float:
    y_pred = model.predict(x_test)
    return accuracy_score(y_test, y_pred)


def predict_image_class(model: svm.SVC, image_path: str) -> list[tuple[str, float]]:
    image = imread(image_path)
    image = resize(image, IMAGE_RESIZE)
    image = image.flatten()
    image = image.reshape(1, -1)
    probabilities = model.predict_proba(image)[0]
    return list(zip(IMAGE_CLASS_NAMES, probabilities))


def main():
    args = parse_args()

    begin_time = time.time()
    x_train, x_test, y_train, y_test = load_data(args.dataset_dir, IMAGE_CLASS_NAMES)
    print(f'Data loaded in {time.time() - begin_time:.2f}s, training data shape: {x_train.shape}, {y_train.shape}')

    begin_time = time.time()
    print('Training SVM')
    model = train_svm(x_train, y_train)
    print(f'Training finished in {time.time() - begin_time:.2f}s')
    
    print('Evaluating SVM')
    accuracy = evaluate_svm(model, x_test, y_test)
    print(f'Accuracy: {accuracy:.4f}')

    print(f'Saving model to {args.model_save_path}')
    with open(args.model_save_path, 'wb') as f:
        pickle.dump(model, f)

if __name__ == '__main__':
    main()
