import imageio
import requests
import bz2
from PIL import Image
import torch
import torchvision.transforms as transforms
import dlib
import os.path

from static.pix2pixHD.data.base_dataset import __scale_width as scale_width
from static.pix2pixHD.models.networks import define_G
import static.pix2pixHD.util.util as util
from static.pix2pixHD.aligner import align_face

import matplotlib.pyplot as plt

from static import config

class GenderTransformer():
    def __init__(self):
        # weights
        self.current_weights = None
        self.weights_paths = {
            "male": config.weights_male,
            "female": config.weights_female
        }

        # models
        self.eval_transform = self._get_eval_transform()
        self.shape_predictor = None
        self.gender_swapper = None

        # you may want to change download flag for true if ypu haven't got any landmarks model
        self._initialize_shape_predictor(download=False)
        self._initialize_gender_swapper()

    def _get_eval_transform(self, loadSize=512):
        """
        create image transformer
        :return: image transformer
        """
        transform_list = []
        transform_list.append(transforms.Lambda(lambda img: scale_width(img,
                                                                          loadSize,
                                                                          Image.BICUBIC)))
        transform_list += [transforms.ToTensor()]
        transform_list += [transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]
        return transforms.Compose(transform_list)

    def _unpack_bz2(self, src_path):
        """
        unpack shape predictor
        :param src_path: path to the dat file
        :return: destination path
        """
        data = bz2.BZ2File(src_path).read()
        dst_path = src_path[:-4]
        with open(dst_path, 'wb') as fp:
            fp.write(data)
        return dst_path

    def _download(self, url, file_name):
        """
        download function for shape predictor
        :param url: url of the model
        :param file_name: download path
        :return: NoneType
        """
        with open(file_name, "wb") as file:
            response = requests.get(url)
            file.write(response.content)

    def _initialize_shape_predictor(self, download=True, shape_model_path = config.landmarks_path):
        """
        init the shape predictor
        :param download: whether to download a model to server
        :param shape_model_path: path to the model
        :return: NoneType
        """

        if download:
            shape_model_url = 'http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2'
            self._download(shape_model_url, shape_model_path)

        self.shape_predictor = dlib.shape_predictor(self._unpack_bz2(shape_model_path))

    def _initialize_gender_swapper(self, load_weights=None):
        """
        initialize pix2pixHD generator
        :param load_weights: 'male'/'female'/None
        :return: NoneType
        """
        config_G = {
            'input_nc': 3,
            'output_nc': 3,
            'ngf': 64,
            'netG': 'global',
            'n_downsample_global': 4,
            'n_blocks_global': 9,
            'n_local_enhancers': 1,
            'norm': 'instance',
        }

        self.gender_swapper = define_G(**config_G)

        if load_weights is not None:
            self._load_weights(load_weights)

    def _load_weights(self, gender: str):
        """
        load the weights for face swap, if needed
        :param gender: can be either "male" or "female"
        :return: NoneType
        """
        if ((self.current_weights is None) or
                (self.current_weights is not None and
                 gender != self.current_weights)):
            try:
                self.gender_swapper.load_state_dict(torch.load(
                    self.weights_paths[gender]
                ))
                self.current_weights = gender
            except Exception as e:
                raise e

    def transform_gender(self, gender, input_path,
                         input_name='image.jpg',
                         output_name_specif='out',
                         output_path=None):
        """
        transform the gender of an image
        :param input_path: path to the image
        :param gender: 'male'/'female'
        :param input_name: name of the image
        :param output_name_specif: 'param_gender.jpg' - name of the output pic
        :param output_path: path to output default input path
        :return: NoneType
        """
        # img_filename = os.path.join(input_path, input_name)
        img_filename = input_path + input_name

        # align face & transform image
        aligned_img = align_face(img_filename, self.shape_predictor)[0]
        # transform = self._get_eval_transform()
        img = self.eval_transform(aligned_img).unsqueeze(0)

        # load weights & swap genders
        self._load_weights(gender)
        with torch.no_grad():
            out = self.gender_swapper(img)

        # save the image
        out = util.tensor2im(out.data[0])
        if output_path is None:
            output_path = input_path
        out_full_path = output_path+output_name_specif+'_'+gender+'.jpg'
        imageio.imsave(out_full_path, out)
