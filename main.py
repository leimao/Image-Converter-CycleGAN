

from model import CycleGAN
from utils import load_data, sample_train_data, image_scaling, image_scaling_inverse

def conversion(model_filepath, img_dir, conversion_direction, output_dir):

    input_size = [256, 256, 3]
    num_filters = 64

    model = CycleGAN(input_size = input_size, num_filters = num_filters)

    model.load(filepath = model_filepath)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for file in os.listdir(img_dir):
        filepath = os.path.join(img_dir, file)
        img = cv2.imread(filepath)
        img = image_scaling(imgs = img)
        img_converted = model.test(inputs = np.array([img]), direction = conversion_direction)[0]
        img_converted = image_scaling_inverse(imgs = img_converted)
        cv2.imwrite(os.path.join(output_dir, os.path.basename(file)), img_converted)


if __name__ == '__main__':

    conversion()