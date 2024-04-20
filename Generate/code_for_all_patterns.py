import os
import cv2
import skimage
import torchvision.transforms as transforms
import albumentations as A
import torch
from PIL import Image, ImageDraw, ImageOps, ImageFilter, ImageEnhance, ImageFont
from augly.image.transforms import *
import random
import augly.image as imaugs
import numpy as np
import math
import imgaug.augmenters as iaa
import imgaug
import legofy
import secrets
import string
from random import getrandbits
import Automold as am
import Helpers as hp
import torchvision
import torchattacks
from dropblock import DropBlock2D
from io import BytesIO
import matplotlib.pyplot as plt
from math import ceil, atan2, sqrt, cos, sin
import pywt
from rembg import remove
import argparse
import pickle

class ToRGB:
    def __call__(self, x):
        return x.convert("RGB")
    
class RandomResizeCrop:
    def __call__(self, x):
        tran = transforms.RandomResizedCrop(256, scale=(0.3,1))
        return tran(x)
    
class Random_image_opaque:
    def __init__(self, opa = [0.2, 1.0], path = 'gsdata/VSC/data/training_images_9/', which = [0,100000]):
        self.opa = opa
        self.path = path
        self.which = which
        self.imgs = os.listdir(self.path)

    def __call__(self, x):
        x = x.convert('RGB')
        opa = random.uniform(self.opa[0], self.opa[1])
        which = random.randint(self.which[0], self.which[1])
        base = Image.open(self.path+self.imgs[which])
        x = x.resize((256,256))
        base = base.resize((256,256))
        assert base.size == x.size
        x = imaugs.overlay_image(base, x, x_pos=0, y_pos=0, opacity=opa)#0.2~1
        return x
    
class Color_jitter():
    def __call__(self, x):
        x = x.convert('RGB')
        x = transforms.ColorJitter(brightness=3, contrast=4, saturation=4, hue=0.5)(x)
        return x
    
class RandomBlur:
    def __init__(self, radius = [2, 5]):
        self.radius = radius
    
    def __call__(self, x):
        x = x.convert('RGB')
        radius = random.uniform(self.radius[0], self.radius[1])
        x = Blur(radius = radius)(x)
        return x
    
class RandomPixelization:
    def __init__(self, ratios = [0.1, 1]):
        self.ratios = ratios
    
    def __call__(self, x):
        x = x.convert('RGB')
        ratio = random.uniform(self.ratios[0], self.ratios[1])
        x = Pixelization(ratio = ratio)(x)
        return x

class RandomRotate:
    def __init__(self, degrees = [0,360]):
        self.degrees = degrees
    
    def __call__(self, x):
        x = x.convert('RGB')
        degree = random.uniform(self.degrees[0], self.degrees[1])
        x = Rotate(degrees = degree)(x)
        return x

class GrayScale:
    def __call__(self, x):
        x = Grayscale()(x)
        return x
    
class RandomPad:
    def __init__(self, w_factors = [0, 0.5], h_factors = [0, 0.5], color_1s = [0,255], color_2s = [0,255], color_3s = [0,255]):
        self.w_factors = w_factors
        self.h_factors = h_factors
        self.color_1s = color_1s
        self.color_2s = color_2s
        self.color_3s = color_3s
    
    def __call__(self, x):
        x = x.convert('RGB')
        w_factor = random.uniform(self.w_factors[0], self.w_factors[1])
        h_factor = random.uniform(self.h_factors[0], self.h_factors[1])
        color_1 = random.randint(self.color_1s[0], self.color_1s[1])
        color_2 = random.randint(self.color_2s[0], self.color_2s[1])
        color_3 = random.randint(self.color_3s[0], self.color_3s[1])
        x = Pad(w_factor = w_factor, h_factor = h_factor, color = (color_1, color_2, color_3))(x)
        return x.resize((256,256))
    
class RandomAddNoise:
    def __init__(self, means = [0, 0.5], varrs = [0, 0.5]):
        self.means = means
        self.varrs = varrs
    def __call__(self, x):
        x = x.convert('RGB')
        mean = random.uniform(self.means[0], self.means[1])
        var = random.uniform(self.varrs[0], self.varrs[1])
        x = RandomNoise(mean = mean, var = var)(x)
        return x
    
class VertFlip:
    def __call__(self, x):
        x = x.convert('RGB')
        return VFlip()(x)
    
class HoriFlip:
    def __call__(self, x):
        x = x.convert('RGB')
        return HFlip()(x)
    
class RandomMemeFormat:
    def __init__(self, text_len = [1, 10], path = 'gsdata/VSC/data/fonts/', opacity = [0, 1], \
                text_colors_0 = [0, 255], text_colors_1 = [0, 255], text_colors_2 = [0, 255], \
                caption_height = [100, 300], \
                bg_colors_0 = [0, 255], bg_colors_1 = [0, 255], bg_colors_2 = [0, 255]):
        self.text_len = text_len
        self.path = path
        self.opacity = opacity
        self.text_colors_0 = text_colors_0
        self.text_colors_1 = text_colors_1
        self.text_colors_2 = text_colors_2
        self.caption_height = caption_height
        self.bg_colors_0 = bg_colors_0
        self.bg_colors_1 = bg_colors_1
        self.bg_colors_2 = bg_colors_2
    
    def __call__(self, x):
        x = x.convert('RGB')
        string = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
        length = random.randint(self.text_len[0], self.text_len[1])
        text = ''.join(random.sample(string, length))
        tiff_path = self.path + random.choice(os.listdir(self.path))
        opacity = random.uniform(self.opacity[0], self.opacity[1])
        text_color_0 = random.randint(self.text_colors_0[0], self.text_colors_0[1])
        text_color_1 = random.randint(self.text_colors_1[0], self.text_colors_1[1])
        text_color_2 = random.randint(self.text_colors_2[0], self.text_colors_2[1])
        height = random.randint(self.caption_height[0], self.caption_height[1])
        bg_color_0 = random.randint(self.bg_colors_0[0], self.bg_colors_0[1])
        bg_color_1 = random.randint(self.bg_colors_1[0], self.bg_colors_1[1])
        bg_color_2 = random.randint(self.bg_colors_2[0], self.bg_colors_2[1])
        x = MemeFormat(text = text,
                       font_file = tiff_path,
                       opacity = opacity,
                       text_color = (text_color_0, text_color_1, text_color_2),
                       caption_height= height,
                       meme_bg_color= (bg_color_0, bg_color_1, bg_color_2))(x)
        return x.resize((256,256))
    
class RandomOverlayEmoji:
    def __init__(self, path = 'gsdata/VSC/data/emoji/', opacity=[0.2, 1], emoji_size=[0.2, 1], x_pos=[0, 0.5], y_pos=[0, 0.5]):
        self.path = path
        self.opacity = opacity
        self.emoji_size = emoji_size
        self.x_pos = x_pos
        self.y_pos = y_pos

    def __call__(self, x):
        x = x.convert('RGB')
        emoji_path = self.path + random.choice(os.listdir(self.path))
        opacity = random.uniform(self.opacity[0], self.opacity[1])
        emoji_size = random.uniform(self.emoji_size[0], self.emoji_size[1])
        x_pos = random.uniform(self.x_pos[0], self.x_pos[1])
        y_pos = random.uniform(self.y_pos[0], self.y_pos[1])
        x = OverlayEmoji(emoji_path = emoji_path,
                         opacity = opacity,
                         emoji_size = emoji_size,
                         x_pos = x_pos,
                         y_pos = y_pos)(x)
        return x
    
class RandomOverlayText(object):
    def __init__(self, text = [0,20], color_1=[0,255], color_2=[0,255], color_3=[0,255], font_size = [0, 1], opacity=[0, 1], x_pos=[0, 0.5], y_pos=[0, 0.5]):
        self.text = text
        self.color_1 = color_1
        self.color_2 = color_2
        self.color_3 = color_3
        self.opacity = opacity
        self.font_size = font_size
        self.x_pos = x_pos
        self.y_pos = y_pos

    def __call__(self, x):
        x = x.convert('RGB')
        text = random.choices(range(100), k = random.randint(self.text[0],self.text[1]))
        color = [random.randint(self.color_1[0],self.color_1[1]),
                 random.randint(self.color_2[0],self.color_2[1]),
                 random.randint(self.color_3[0],self.color_3[1])]
        opacity = random.uniform(self.opacity[0], self.opacity[1])
        font_size = random.uniform(self.font_size[0], self.font_size[1])
        x_pos = random.uniform(self.x_pos[0], self.x_pos[1])
        y_pos = random.uniform(self.y_pos[0], self.y_pos[1])
        x = OverlayText(text = text,
                        font_size = font_size,
                        opacity = opacity,
                        color = color,
                        x_pos = x_pos,
                        y_pos = y_pos)(x)
        return x
    
class RandomPerspectiveTransform:
    def __init__(self, sigmas = [10, 100]):
        self.sigmas = sigmas
    def __call__(self, x):
        x = x.convert('RGB')
        sigma = random.uniform(self.sigmas[0], self.sigmas[1])
        x = PerspectiveTransform(sigma=sigma)(x)
        return x
    
class RandomOverlayImage:
    def __init__(self, path = 'gsdata/VSC/data/training_images_9//', opacity=[0.6, 1], overlay_size=[0.5, 1], x_pos=[0, 0.5], y_pos=[0, 0.5]):
        self.path = path
        self.opacity = opacity
        self.overlay_size = overlay_size
        self.x_pos = x_pos
        self.y_pos = y_pos

    def __call__(self, x):
        x = x.convert('RGB')
        bg = Image.open(self.path + random.choice(os.listdir(self.path)))
        opacity = random.uniform(self.opacity[0], self.opacity[1])
        overlay_size = random.uniform(self.overlay_size[0], self.overlay_size[1])
        x_pos = random.uniform(self.x_pos[0], self.x_pos[1])
        y_pos = random.uniform(self.y_pos[0], self.y_pos[1])
        bg = OverlayImage(overlay = x,
                         opacity = opacity,
                         overlay_size = overlay_size,
                         x_pos = x_pos,
                         y_pos = y_pos)(bg)
        return bg.resize((256,256))
    
class RandomStackImage:
    def __init__(self, path = 'gsdata/VSC/data/training_images_9/', choice_1 = [0, 1], choice_2 = [0, 1]):
        self.path = path
        self.choice_1 = choice_1
        self.choice_2 = choice_2

    def __call__(self, x):
        x = x.convert('RGB')
        bg = Image.open(self.path + random.choice(os.listdir(self.path)))
        choice_1 = random.randint(self.choice_1[0],self.choice_1[1])
        choice_2 = random.randint(self.choice_2[0],self.choice_2[1])
        
        if choice_1 == 0:
            image1 = x.resize((256,256))
            image2 = bg.resize((256,256))
        else:
            image1 = bg.resize((256,256))
            image2 = x.resize((256,256))
        
        if choice_2 ==0:
            new_image = Image.new('RGB', (image1.width + image2.width, max(image1.height, image2.height)))
            new_image.paste(image1, (0, 0))
            new_image.paste(image2, (image1.width, 0))
            new_image = new_image.resize((256,256))
        else:
            new_image = Image.new('RGB', (max(image1.width, image2.width), image1.height + image2.height))
            new_image.paste(image1, (0, 0))
            new_image.paste(image2, (0, image1.height))
            new_image = new_image.resize((256,256))
        return new_image
    
class RandomChangeChannel(object):
    def __init__(self, shift_v = [0,40], order_v = [0,1,2], invert_0 = [0,1], invert_1 = [0,1], invert_2 = [0,1]):
        self.shift_v = shift_v
        self.order_v = order_v
        self.invert_0 = invert_0
        self.invert_1 = invert_1
        self.invert_2 = invert_2

    def shift_channels(self, image, shift_v=[20, -20]):
        image = np.array(image) 
        if len(image.shape) == 3 and image.shape[-1] == 3:
            H_i,W_i,C_i = image.shape
            I, J = shift_v

            image[:,:,0] = np.roll(image[:,:,0], (I, J), axis=(0,1) )
            image[:,:,2] = np.roll(image[:,:,2], (-I, -J), axis=(0,1) )

            I, J = abs(I), abs(J)
            if I>0 and J>0:
                image = image[I:-I,J:-J] 
            elif I==0 and J>0:
                image = image[:,J:-J] 
            elif I>0 and J==0:
                image = image[I:-I] 
        return Image.fromarray(image)
    
    def swap_channels(self, image, new_channel_order_v=[2,1,0]):
        image = np.array(image) 
        if len(image.shape) == 3 and image.shape[-1] == 3:
            image = image[:,:,new_channel_order_v]

        return Image.fromarray(image)
    
    def invert_channel(self, image, invert_r=False, invert_g=False,invert_b=True):
        image = np.array(image) 
        if len(image.shape) == 3 and image.shape[-1] == 3:
            if invert_r:
                image[:,:,0] = 255 - image[:,:,0]

            if invert_g:
                image[:,:,1] = 255 - image[:,:,1]

            if invert_b:
                image[:,:,2] = 255 - image[:,:,2]
        return Image.fromarray(image) 

    def __call__(self, x):
        x = x.convert('RGB')
        
        select = random.choices(range(3), k=1)[0]
        if select==0:
            sv = int(random.uniform(self.shift_v[0], self.shift_v[1]))
            x = self.shift_channels(x, shift_v=[sv, -sv])
        if select==1:
            ls = [0,1,2]
            random.shuffle(ls)
            x = self.swap_channels(x, new_channel_order_v=ls)
        if select==2:
            invert_r = random.randint(self.invert_0[0],self.invert_0[1])
            invert_g = random.randint(self.invert_1[0],self.invert_1[1])
            invert_b = random.randint(self.invert_2[0],self.invert_2[1])
            x = self.invert_channel(x, invert_r=invert_r, invert_g=invert_g, invert_b=invert_b)
        
        return x
    
class RandomEncodingQuality(object):
    def __init__(self, quality = [0, 50]):
        self.quality = quality
    def __call__(self, x):
        x = x.convert('RGB')
        quality = int(random.uniform(self.quality[0], self.quality[1]))
        x = EncodingQuality(quality=quality)(x)
        return x
    
class RandomOverlayStripes(object):
    def __init__(self, line_widths = [0, 1], \
                 line_color_0 = [0, 255], line_color_1 = [0, 255], line_color_2 = [0, 255], \
                 line_angles = [0, 360], line_densitys = [0, 1], line_opacitys = [0, 1]):
        self.line_widths = line_widths
        self.line_color_0 = line_color_0
        self.line_color_1 = line_color_1
        self.line_color_2 = line_color_2
        self.line_angles = line_angles
        self.line_densitys = line_densitys
        self.line_opacitys = line_opacitys

    def __call__(self, x):
        x = x.convert('RGB')
        line_width = random.uniform(self.line_widths[0], self.line_widths[1])
        line_color = (random.randint(self.line_color_0[0], self.line_color_0[1]), \
                       random.randint(self.line_color_1[0], self.line_color_1[1]), \
                       random.randint(self.line_color_2[0], self.line_color_2[1]))
        line_angle = random.randint(self.line_angles[0], self.line_angles[1])
        line_density = random.uniform(self.line_densitys[0], self.line_densitys[1])
        line_opacity = random.uniform(self.line_opacitys[0], self.line_opacitys[1])

        x = OverlayStripes(line_width = line_width, \
                           line_color = line_color, \
                           line_angle = line_angle, \
                           line_density = line_density, \
                           line_opacity = line_opacity)(x)

        return x
    
class RandomSharpen(object):
    def __init__(self, factors = [2, 20]):
        self.factors = factors

    def __call__(self, x):
        x = x.convert('RGB')
        factor = random.uniform(self.factors[0], self.factors[1])
        x = Sharpen(factor = factor)(x)
        return x
    
class RandomSkew(object):
    def __init__(self, skew_factors = [-2, 2]):
        self.skew_factors = skew_factors

    def __call__(self, x):
        x = x.convert('RGB')
        skew_factor = random.uniform(self.skew_factors[0], self.skew_factors[1])
        x = Skew(skew_factor = skew_factor)(x)
        return x
    
class RandomShufflePixels(object):
    def __init__(self, factors = [0.1, 0.5]):
        self.factors = factors

    def __call__(self, x):
        x = x.convert('RGB')
        factor = random.uniform(self.factors[0], self.factors[1])
        x = ShufflePixels(factor = factor)(x)
        return x
    
class RandomAddShapes(object):
    def __init__(self, num = [50,500], \
                 color_0 = [0, 255], color_1 = [0, 255], color_2 = [0, 255], \
                 radius = [1, 30], \
                 shape_types = ["ellipse", "rectangle", "triangle", "star", "pentagon"]):
        self.num = num
        self.color_0 = color_0
        self.color_1 = color_1
        self.color_2 = color_2
        self.radius = radius
        self.shape_types = shape_types
        
    def draw_star(self, draw, center, radius, fill=None):
        angle = 2 * math.pi / 5
        outer_radius = radius
        inner_radius = radius / 2.0

        points = []
        for i in range(10):
            if i % 2 == 0:
                r = outer_radius
            else:
                r = inner_radius
            x = center[0] + r * math.cos(i * angle)
            y = center[1] - r * math.sin(i * angle)
            points.extend((x, y))
        draw.polygon(points, fill=fill)

    def draw_pentagon(self, draw, center, radius, fill=None):
        angle = 2 * math.pi / 5
        points = []
        for i in range(5):
            x = center[0] + radius * math.cos(i * angle)
            y = center[1] - radius * math.sin(i * angle)
            points.extend((x, y))
        draw.polygon(points, fill=fill)

    def __call__(self, image):
        image = image.convert('RGB')
        
        color = (random.randint(self.color_0[0], self.color_0[1]), \
                       random.randint(self.color_1[0], self.color_1[1]), \
                       random.randint(self.color_2[0], self.color_2[1]))
        num = random.randint(self.num[0], self.num[1])
        radius = random.uniform(self.radius[0], self.radius[1])
        shape_type = random.choice(self.shape_types)
        
        img_1 = image.copy()
        draw = ImageDraw.Draw(img_1)
        width, height = image.size
        
        for _ in range(num):
            x = random.randint(0, width)
            y = random.randint(0, height)
            
            if shape_type == "ellipse":
                draw.ellipse([(x - radius, y - radius), (x + radius, y + radius)], fill=color)
            elif shape_type == "rectangle":
                draw.rectangle([(x - radius, y - radius), (x + radius, y + radius)], fill=color)
            elif shape_type == "triangle":
                points = [(x, y - radius), (x + radius, y + radius), (x - radius, y + radius)]
                draw.polygon(points, fill=color)
            elif shape_type == "star":
                self.draw_star(draw, (x, y), radius, fill=color)
            elif shape_type == "pentagon":
                self.draw_pentagon(draw, (x, y), radius, fill=color)
            
        return img_1
    
class RandomRepeat(object):
    def __init__(self, repeat_horizontal = 4, repeat_vertical = 4):
        self.repeat_horizontal = repeat_horizontal
        self.repeat_vertical = repeat_vertical
    
    def __call__(self, image):
        image = image.convert('RGB')
        
        repeat_horizontal = random.randint(1,self.repeat_horizontal)
        repeat_vertical = random.randint(1,self.repeat_vertical)
        
        image = image.resize((256,256))
        width, height = image.size
        new_width = width * repeat_horizontal
        new_height = height * repeat_vertical
        new_image = Image.new('RGB', (new_width, new_height))

        for i in range(repeat_horizontal):
            for j in range(repeat_vertical):
                new_image.paste(image, (i * width, j * height))

        return new_image.resize((256,256))
    
class RandomCutAssemble(object):
    def __init__(self, num_cuts_horizontal = 5, num_cuts_vertical = 5):
        self.num_cuts_horizontal = num_cuts_horizontal
        self.num_cuts_vertical = num_cuts_vertical
    
    def __call__(self, image):
        image = image.convert('RGB')
        
        num_cuts_horizontal = random.randint(1,self.num_cuts_horizontal)
        num_cuts_vertical = random.randint(1,self.num_cuts_vertical)
        
        
        width, height = image.size
        cell_width = width // num_cuts_horizontal
        cell_height = height // num_cuts_vertical

        cells = []
        for i in range(num_cuts_horizontal):
            for j in range(num_cuts_vertical):
                left = i * cell_width
                upper = j * cell_height
                right = left + cell_width
                lower = upper + cell_height
                cell = image.crop((left, upper, right, lower))
                cells.append(cell)

        random.shuffle(cells)
        new_image = Image.new('RGB', (width, height))
        index = 0
        for i in range(num_cuts_horizontal):
            for j in range(num_cuts_vertical):
                left = i * cell_width
                upper = j * cell_height
                new_image.paste(cells[index], (left, upper))
                index += 1

        return new_image
    
class BGRemovePaste(object):
    def __init__(self, opa = [0.5, 1.0], x_pos = [0, 0.5], y_pos = [0, 0.5], \
                 path = 'gsdata/VSC/data/training_images_9/', which = [0,100000]):
        self.opa = opa
        self.path = path
        self.which = which
        self.imgs = os.listdir(self.path)
        self.x_pos = x_pos
        self.y_pos = y_pos

    def __call__(self, x):
        x = x.convert('RGB')
        opa = random.uniform(self.opa[0], self.opa[1])
        which = random.randint(self.which[0], self.which[1])
        base = Image.open(self.path+self.imgs[which])
        x_pos = random.uniform(self.x_pos[0], self.x_pos[1])
        y_pos = random.uniform(self.y_pos[0], self.y_pos[1])
        x = remove(x)
        x = x.resize((256,256))
        base = base.resize((256,256))
        x = imaugs.overlay_image(base, x, x_pos=x_pos, y_pos=y_pos, opacity=opa)#0.2~1
        return x.convert('RGB')
    
class Xraylize(object):
    def __call__(self, image):
        image = image.convert('RGB')
        greyscale_image = image.convert('L')
        inverted_image = ImageOps.invert(greyscale_image)
        return inverted_image
    
class Mirrorize(object):
    def __init__(self, choices=[0,1]):
        self.choices = choices
    
    def __call__(self, image):
        image = image.convert('RGB')
        choice = random.randint(self.choices[0], self.choices[1])
        
        if choice ==0:
            flipped_image = image.transpose(Image.FLIP_LEFT_RIGHT)
            width, height = image.size
            new_width = width * 2
            mirror_image = Image.new('RGB', (new_width, height))
            mirror_image.paste(image, (0, 0))
            mirror_image.paste(flipped_image, (width, 0))
            return mirror_image.resize((256,256))
        
        else:
            flipped_image = image.transpose(Image.FLIP_TOP_BOTTOM)
            width, height = image.size
            new_height = height * 2
            mirror_image = Image.new('RGB', (width, new_height))
            mirror_image.paste(image, (0, 0))
            mirror_image.paste(flipped_image, (0, height))

            return mirror_image.resize((256,256))
        
class Kaleidoscope(object):
    def __init__(self, num_slices=10):
        self.num_slices = num_slices
    
    def create_pie_slice(self, image, angle):
        width, height = image.size
        diagonal = int((width ** 2 + height ** 2) ** 0.5)
        mask = Image.new('1', (diagonal, diagonal), 0)
        draw = ImageDraw.Draw(mask)

        draw.pieslice((0, 0, diagonal, diagonal), 0, angle, fill=1)
        mask = mask.resize(image.size, resample=Image.LANCZOS)

        pie_slice = ImageOps.fit(image, mask.size, centering=(0.5, 0.5))
        pie_slice.putalpha(mask)

        return pie_slice
    
    def kaleidoscope(self, image, num_slices):
        width, height = image.size

        # Create a pie slice of the image
        angle = 360 // num_slices
        pie_slice = self.create_pie_slice(image, angle)

        # Create the kaleidoscope image
        kaleidoscope_image = Image.new('RGBA', (width, height), (0, 0, 0, 0))

        for i in range(num_slices):
            rotated_slice = pie_slice.rotate(-angle * i)
            kaleidoscope_image.paste(rotated_slice, (0, 0), rotated_slice)

        return kaleidoscope_image
    
    def __call__(self, image):
        image = image.convert('RGB')
        num_slice = random.randint(1, self.num_slices)
        k_image = self.kaleidoscope(image, num_slice)
        return k_image
    
class EdgeDetection(object):
    def __init__(self, choices=4):
        self.choices = choices
        
        
    def __call__(self, image):
        image = image.convert('RGB')
        choice = random.randint(0, self.choices)
        if choice == 0:
            grayscale_image = image.convert('L')
            edge_image = grayscale_image.filter(ImageFilter.FIND_EDGES)
            return edge_image
        elif choice == 1:
            image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            canny_image = cv2.Canny(gray_image, 100, 200)
            return Image.fromarray(canny_image)
        elif choice == 2:
            image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
            sobel_image = cv2.addWeighted(sobel_x, 0.5, sobel_y, 0.5, 0)
            sobel_image = cv2.convertScaleAbs(sobel_image)
            return Image.fromarray(sobel_image)
        elif choice == 3:
            matrix = np.array([[0, -1, 0],
                               [-1, 4, -1],
                               [0, -1, 0]])
            aug = iaa.Convolve(matrix=matrix)
            return Image.fromarray(aug(images = [np.array(image)])[0])
        elif choice == 4:
            aug = iaa.pillike.FilterContour()
            return Image.fromarray(aug(images = [np.array(image)])[0])
        
class Glass(object):
    def __init__(self, radius=[3,10]):
        self.radius = radius
    
    def __call__(self, image):
        image = image.convert('RGB')
        radius = random.randint(self.radius[0], self.radius[1])
        
        width, height = image.size
        glass_image = Image.new('RGB', (width, height))

        for x in range(width):
            for y in range(height):
                # Get a random offset within the specified radius
                offset_x = random.randint(-radius, radius)
                offset_y = random.randint(-radius, radius)

                # Get the source pixel coordinates
                src_x = min(max(x + offset_x, 0), width - 1)
                src_y = min(max(y + offset_y, 0), height - 1)

                # Copy the source pixel to the destination pixel
                pixel = image.getpixel((src_x, src_y))
                glass_image.putpixel((x, y), pixel)

        return glass_image
    
class OpticalDistortion(object):
    def __init__(self,distort_limit=[2,10], shift_limit=[0,1], interpolation=[0,4], border_mode=[0,5], value=[0,255],always_apply=False, p=1):
        self.distort_limit = distort_limit
        self.shift_limit = shift_limit
        self.interpolation = interpolation
        self.border_mode = border_mode
        self.value = value
        self.always_apply = always_apply
        self.p = p
    
    def __call__(self, img):
        img = img.convert('RGB')
        img = np.array(img)
        distort_limit = random.uniform(self.distort_limit[0],self.distort_limit[1])
        shift_limit = random.uniform(self.shift_limit[0],self.shift_limit[1])
        interpolation = random.randint(self.interpolation[0],self.interpolation[1])
        border_mode = random.randint(self.border_mode[0],self.border_mode[1])
        value = random.randint(self.value[0],self.value[1])
        transform = A.OpticalDistortion(distort_limit = distort_limit, \
                                        shift_limit = shift_limit, \
                                        interpolation = interpolation, \
                                        border_mode = border_mode, \
                                        value = value, \
                                        always_apply = self.always_apply, \
                                        p = self.p)
        new_img = transform(image=img)['image']
        new_img = Image.fromarray(new_img)
        return new_img
    
class WeatherChange(object):
    def __init__(self, choices = 5, snow_coeff = [0.1, 1], frost_ser = [1,5]):
        self.choices = choices
        self.snow_coeff = snow_coeff
        self.frost_ser = frost_ser
        
    def __call__(self, image):
        image = image.convert('RGB')
        choice = random.randint(0, self.choices)
        if choice == 0:
            aug = iaa.Rain(nb_iterations=(5,10), drop_size=(0.01, 0.02), speed=(0.01,0.03))    
        elif choice == 1:
            aug = iaa.Fog()
        elif choice == 2:
            aug = iaa.Clouds()
        elif choice == 3:
            aug = iaa.Snowflakes(density=(0.3,0.5), flake_size=(0.7, 0.95), speed=(0.001, 0.01))
        elif choice == 4:
            snow_coeff = random.uniform(self.snow_coeff[0], self.snow_coeff[1])
            snowy_images= am.add_snow(hp.load_image(image), snow_coeff=snow_coeff)
            return Image.fromarray(cv2.cvtColor(snowy_images, cv2.COLOR_BGR2RGB))
        elif choice == 5:
            frost_ser = random.randint(self.frost_ser[0], self.frost_ser[1])
            aug = iaa.imgcorruptlike.Frost(severity=frost_ser)
            return Image.fromarray(aug(images=[np.array(image)])[0])
        if choice != 4:
            return Image.fromarray(aug.augment_image(np.array(image)))
        
class RandomSplitRotate(object):
    def __init__(self,choice = [0,1],x_split = [0,1],y_split = [0,1],degree = [0,360]):
        self.choice = choice
        self.x_split = x_split
        self.y_split = y_split
        self.degree = degree
    
    def __call__(self,img):
        img = img.convert('RGB')
        choice = random.randint(self.choice[0],self.choice[1])
        if choice == 0:
            x_split = random.uniform(self.x_split[0],self.x_split[1])
            img_np = np.array(img)
            width = img.size[0]
            split_pos = int(width*x_split)
            image_1 = img_np[:,:split_pos]
            image_2 = img_np[:,split_pos:-1]
            image_1 = Image.fromarray(image_1)
            image_2 = Image.fromarray(image_2)
            degree_1 = random.uniform(self.degree[0],self.degree[1])
            degree_2 = random.uniform(self.degree[0],self.degree[1])
            image_1 = Rotate(degrees=degree_1)(image_1)
            image_2 = Rotate(degrees=degree_2)(image_2)
            new_image = Image.new('RGB', (image_1.width + image_2.width, max(image_1.height, image_2.height)))
            new_image.paste(image_1, (0, 0))
            new_image.paste(image_2, (image_1.width, 0))
            new_image = new_image.resize((256,256))
        else:
            y_split = random.uniform(self.y_split[0],self.y_split[1])
            img_np = np.array(img)
            width = img.size[1]
            split_pos = int(width*y_split)
            image_1 = img_np[:split_pos,:]
            image_2 = img_np[split_pos:-1,:]
            image_1 = Image.fromarray(image_1)
            image_2 = Image.fromarray(image_2)
            degree_1 = random.uniform(self.degree[0],self.degree[1])
            degree_2 = random.uniform(self.degree[0],self.degree[1])
            image_1 = Rotate(degrees=degree_1)(image_1)
            image_2 = Rotate(degrees=degree_2)(image_2)            
            new_image = Image.new('RGB', (image_1.width + image_2.width, max(image_1.height, image_2.height)))
            new_image.paste(image_1, (0, 0))
            new_image.paste(image_2, (image_1.width, 0))
            new_image = new_image.resize((256,256))

        return new_image
    
class Solarize(object):
    def __init__(self, thresholds=[0,100]):
        self.thresholds = thresholds
        
    def __call__(self, image):
        image = image.convert('RGB')
        threshold = random.randint(self.thresholds[0], self.thresholds[1])
        solarized_image = ImageOps.solarize(image, threshold)
        return solarized_image
    
class Legofy(object):
    def __init__(self, thresholds=[5,20]):
        self.thresholds = thresholds
    
    def __call__(self, image):
        image = image.convert('RGB')
        threshold = random.randint(self.thresholds[0], self.thresholds[1])
        alphabet = string.ascii_letters + string.digits
        password = ''.join(secrets.choice(alphabet) for i in range(16))
        image.save('/tmp/' +  password + '.png', quality=100)
        legofy.main('/tmp/' +  password + '.png', output_path= '/tmp/' +  password + '.png', \
            palette_mode='all', size=int(max(image.size)/threshold))
        legofy_image = Image.open('/tmp/' +  password + '.png')
        os.remove('/tmp/' +  password + '.png')
        return legofy_image.resize((256,256))
    
class Ink(object):
    def __init__(self,choice = [0,1], blur_radius = [5,20]):
        self.choice = choice
        self.blur_radius = blur_radius
        
    def __call__(self, image):
        image = image.convert('RGB')
        choice = random.randint(self.choice[0],self.choice[1]) 
        blur_radius = random.randint(self.blur_radius[0], self.blur_radius[1]) 
        if choice == 0:
            a=np.asarray(image.convert("L")).astype("float")
            depth=10
            grad = np.gradient(a) 
            grad_x, grad_y = grad   
            grad_x = grad_x*depth/100.
            grad_y = grad_y*depth/100.
            AA = np.sqrt(grad_x**2 + grad_y**2 + 1.)
            uni_x = grad_x/AA
            uni_y = grad_y/AA
            uni_z = 1./AA
            vec_el = np.pi/2.2                   
            vec_az = np.pi/4.                     
            dx = np.cos(vec_el)*np.cos(vec_az)   
            dy = np.cos(vec_el)*np.sin(vec_az)   
            dz = np.sin(vec_el)           
            b = 255*(dx*uni_x + dy*uni_y + dz*uni_z)   
            b = b.clip(0,255)
            ink_painting_image = Image.fromarray(b.astype('uint8'))  
        else:
            grayscale_image = image.convert('L')
            blurred_image = grayscale_image.filter(ImageFilter.GaussianBlur(blur_radius))
            inverted_image = ImageOps.invert(blurred_image)
            ink_painting_image = Image.blend(grayscale_image, inverted_image, alpha=0.5)
        return ink_painting_image
    
class ZoomBlur(object):
    def __init__(self,max_factor=1.31, step_factor=(0.01, 0.03),always_apply=False, p=1):
        self.max_factor = max_factor
        self.step_factor = step_factor
        self.always_apply = always_apply
        self.p = p
    
    def __call__(self,img):
        img = img.convert('RGB')
        img = np.array(img)
        transform = A.ZoomBlur(max_factor=self.max_factor,step_factor=self.step_factor,always_apply=self.always_apply,p=self.p)
        new_img = transform(image=img)['image']
        new_img = Image.fromarray(new_img)
        return new_img
    
class RandomCutPaste(object):
    def __init__(self, which = [0, 1], factors_w = [0.1, 0.5], factors_h = [0.1, 0.5], color_1s = [0,255], color_2s = [0,255], color_3s = [0,255]):
        self.which = which
        self.factors_w = factors_w
        self.factors_h = factors_h
        self.color_1s = color_1s
        self.color_2s = color_2s
        self.color_3s = color_3s
        
    
    def __call__(self, image):
        image = image.convert('RGB')
        which = random.randint(self.which[0], self.which[1]) 
        factor_w = random.uniform(self.factors_w[0], self.factors_w[1]) 
        factor_h = random.uniform(self.factors_h[0], self.factors_h[1]) 
        color_1 = random.randint(self.color_1s[0], self.color_1s[1])
        color_2 = random.randint(self.color_2s[0], self.color_2s[1])
        color_3 = random.randint(self.color_3s[0], self.color_3s[1])
        
        width, height = image.size
        
        cut_width = int(width * factor_w)
        cut_height = int(height * factor_h)
        
        img_copy = image.copy()
        draw = ImageDraw.Draw(img_copy)

        cut_x = random.randint(0, width - cut_width)
        cut_y = random.randint(0, height - cut_height)

        cut_region = img_copy.crop((cut_x, cut_y, cut_x + cut_width, cut_y + cut_height))

        paste_x = random.randint(0, width - cut_width)
        paste_y = random.randint(0, height - cut_height)

        paste_region = img_copy.crop((paste_x, paste_y, paste_x + cut_width, paste_y + cut_height))
        
        draw.rectangle([cut_x, cut_y, cut_x + cut_width, cut_y + cut_height], fill=(color_1, color_2, color_3))

        img_copy.paste(cut_region, (paste_x, paste_y))
        if which == 1:
            img_copy.paste(paste_region, (cut_x, cut_y))
        
        return img_copy
    
class Cartoonize(object):
    def __call__(self, image):
        image = image.convert('RGB')
        image = cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2BGR)
        # Resize the image
        height, width, _ = image.shape
        image = cv2.resize(image, (800, int(800 * height / width)), interpolation=cv2.INTER_AREA)

        # Convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply median blur to the grayscale image
        gray_blurred = cv2.medianBlur(gray, 7)

        # Detect edges using adaptive thresholding
        edges = cv2.adaptiveThreshold(gray_blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)

        # Convert the image to a color cartoon
        color_cartoon = cv2.bilateralFilter(image, 9, 300, 300)
        for _ in range(2):
            color_cartoon = cv2.bilateralFilter(color_cartoon, 9, 300, 300)

        # Combine the color cartoon image and the edges
        cartoon_image = cv2.bitwise_and(color_cartoon, color_cartoon, mask=edges)
        cartoon_image = Image.fromarray(cv2.cvtColor(cartoon_image, cv2.COLOR_BGR2RGB)) 
        return cartoon_image

class FDA(object):
    '''
    https://github.com/YanchaoYang/FDA
    '''
    def __init__(self,reference_images = 'gsdata/VSC/data/emoji/', beta_limit=0.1, read_fn=lambda x: x,always_apply=False, p=1):
        self.reference_images = os.listdir(reference_images)
        self.beta_limit = beta_limit
        self.read_fn = read_fn
        self.always_apply = always_apply
        self.p = p
    
    def __call__(self,img):
        img = img.convert('RGB')
        img = np.array(img)
        target_img = Image.open('gsdata/VSC/data/emoji/' + random.choice(self.reference_images))
        target_img = target_img.convert('RGB')
        target_img = np.array(target_img)
        transform = A.FDA(reference_images=[target_img],beta_limit=self.beta_limit,read_fn=self.read_fn,always_apply=self.always_apply,p=self.p)
        new_img = transform(image=img)['image']
        new_img = Image.fromarray(new_img)
        return new_img
    
class OilPainting(object):
    def __init__(self, templateSizes=[1,1], steps=[2,2]):
        self.templateSizes = templateSizes
        self.steps = steps
        
        
    def paint(self, img, templateSize, bucketSize, step):
        gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
        gray = ((gray/256)*bucketSize).astype(int)                          #灰度图在桶中的所属分区
        h,w = img.shape[:2]

        oilImg = np.zeros(img.shape, np.uint8)                              #用来存放过滤图像

        for i in range(0,h,step):

            top = i-templateSize
            bottom = i+templateSize+1
            if top < 0:
                top = 0
            if bottom >= h:
                bottom = h-1

            for j in range(0,w,step):

                left = j-templateSize
                right = j+templateSize+1
                if left < 0:
                    left = 0
                if right >= w:
                    right = w-1

                # 灰度等级统计
                buckets = np.zeros(bucketSize,np.uint8)                     #桶阵列，统计在各个桶中的灰度个数
                bucketsMean = [0,0,0]                                       #对像素最多的桶，求其桶中所有像素的三通道颜色均值
                #对模板进行遍历
                for c in range(top,bottom):
                    for r in range(left,right):
                        buckets[gray[c,r]] += 1                         #模板内的像素依次投入到相应的桶中，有点像灰度直方图

                maxBucket = np.max(buckets)                                 #找出像素最多的桶以及它的索引
                maxBucketIndex = np.argmax(buckets)

                for c in range(top,bottom):
                    for r in range(left,right):
                        if gray[c,r] == maxBucketIndex:
                            bucketsMean += img[c,r]
                bucketsMean = (bucketsMean/maxBucket).astype(int)           #三通道颜色均值

                # 油画图
                for m in range(step):
                    for n in range(step):
                        oilImg[m+i,n+j] = (bucketsMean[0],bucketsMean[1],bucketsMean[2])
        return Image.fromarray(cv2.cvtColor(oilImg, cv2.COLOR_BGR2RGB))
    
    def __call__(self, image):
        image = image.convert('RGB')
        image = image.resize((256,256))
        img = cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2BGR)
        
        
        templateSize = random.randint(self.templateSizes[0], self.templateSizes[1])
        bucketSize = 8
        step = random.randint(self.steps[0], self.steps[1])
        success = False
        while not success:
            try:
                oil_image = self.paint(img, templateSize, bucketSize, step)
                success = True
            except:
                success = False
        return oil_image
    
class Piece(object):
    def __init__(self, image, x, y, jigsaw):
        self.image = image
        self.w, self.h = self.image.size
        self.x = x
        self.y = y
        self.jigsaw = jigsaw

    def __repr__(self):
        return '%sx%s' % (self.x, self.y)

    @property
    def filename(self):
        return '%sx%s.png' % (self.x, self.y)

    def save(self, output):
        self.image.save('%s/%s' % (output, self.filename))

    def left(self):
        if self.x == 0:
            return False
        return self.jigsaw.get(self.x - 1, self.y)

    def up(self):
        if self.y == 0:
            return False
        return self.jigsaw.get(self.x, self.y - 1)

    def right(self):
        if self.x == self.jigsaw.piece_count_w - 1:
            return False
        return self.jigsaw.get(self.x + 1, self.y)

    def down(self):
        if self.y == self.jigsaw.piece_count_h - 1:
            return False
        return self.jigsaw.get(self.x, self.y + 1)

    def copy(self, polygon, x=0, y=0):
        polygon, box, coords = self._calculate_polygon(polygon, x, y)
        im_array = np.asarray(self.image)
        mask_im = Image.new('L', (im_array.shape[1], im_array.shape[0]), 0)
        drw = ImageDraw.Draw(mask_im)
        drw.polygon(polygon, outline=1, fill=1)
        mask = np.array(mask_im)

        new_im_array = np.empty(im_array.shape, dtype='uint8')

        new_im_array[:, :, :3] = im_array[:, :, :3]

        new_im_array[:, :, 3] = mask * 255

        piece = Image.fromarray(new_im_array, 'RGBA')
        piece = piece.crop(box)
        return piece

    def cut(self, polygon, x=0, y=0):
        polygon, box, coords = self._calculate_polygon(polygon, x, y)
        piece = self.copy(polygon)
        drw = ImageDraw.Draw(self.image)
        drw.polygon(polygon, fill=(255, 255, 255, 0))
        return piece

    def paste(self, piece, polygon, x=0, y=0):
        polygon, box, coords = self._calculate_polygon(polygon, x, y)
        self.image.paste(piece, box)

    def draw_merge_piece(self, polygon, other, x=0, y=0, other_x=0, other_y=0):
        merge_piece = self.cut(polygon, x, y)
        if other:
            other.paste(merge_piece, polygon, other_x, other_y)

    def create_connections(self):
        w, h = self.image.size
        m = int(self.jigsaw.margin)
        hw = w/2
        hh = h/2
        polygon = [(0, 0), (m, 0), (m, m), (0, m)]

        other = self.left()
        if other:
            if bool(getrandbits(1)):
                self.draw_merge_piece(polygon, other, x=int(m), y=int(hh-m/2),
                                      other_x=int(w-m), other_y=int(hh-m/2))
            else:
                other.draw_merge_piece(polygon, self, x=int(w-m*2), y=int(hh-m/2),
                                       other_x=int(0), other_y=int(hh-m/2))

        other = self.up()
        if other:
            if bool(getrandbits(1)):
                self.draw_merge_piece(polygon, other, x=int(hw-m/2), y=int(m),
                                      other_x=int(hw-m/2), other_y=int(h-m))
            else:
                other.draw_merge_piece(polygon, self, x=int(hw-m/2), y=int(h-m*2),
                                       other_x=int(hw-m/2), other_y=0)

    def _calculate_polygon(self, polygon, x=0, y=0):
        polygon = [(i[0] + x, i[1] + y) for i in polygon]
        box = (min([i[0] for i in polygon]),
               min([i[1] for i in polygon]),
               max([i[0] for i in polygon]),
               max([i[1] for i in polygon]))
        coords = (box[0], box[1])
        return polygon, box, coords
    
class Jigsaw(object):
    def __init__(self, image, output, piece_count_w, piece_count_h):
        self.image = image
        self.image_width, self.image_height = self.image.size
        self.piece_width = int(self.image_width / piece_count_w)
        self.piece_height = int(self.image_height / piece_count_h)
        self.output = output
        self.piece_count_w = piece_count_w
        self.piece_count_h = piece_count_h
        self.margin = (self.piece_width + self.piece_height) / 2 / 5
        self.rows = []

        for y in range(self.piece_count_h):
            row = []
            for x in range(self.piece_count_w):
                left = x * self.piece_width
                up = y * self.piece_height
                right = left + self.piece_width
                bottom = up + self.piece_height
                box = (left, up, right, bottom)
                piece = self.image.crop(box)

                new_width, new_height = map(lambda x: x + self.margin * 2,
                                            piece.size)

                new_im = Image.new('RGBA', (int(new_width), int(new_height)))
                portions = (int(self.margin),
                            int(self.margin),
                            int(self.piece_width + self.margin),
                            int(self.piece_height + self.margin))

                transparent_area = (0, 0, new_width, new_height)
                draw = ImageDraw.Draw(new_im)
                draw.rectangle(transparent_area, fill=0)

                new_im.paste(piece, portions)
                row.append(Piece(new_im, x, y, self))
            self.append(row)
        for row in self.rows:
            for piece in row:
                piece.create_connections()
        self.save()

    def __repr__(self):
        text = ""
        for row in self.rows:
            for piece in row:
                text += '[%s]' % piece
            text += '\n'
        return text

    def save(self):
        for row in self.rows:
            for piece in row:
                piece.save(self.output)

    def get(self, x, y):
        return self.rows[y][x]

    def append(self, row):
        self.rows.append(row)

    def get_pieces(self):
        rows = []
        for row in self.rows:
            columns = []
            for piece in row:
                columns.append({'x': piece.x, 'y': piece.y})
            rows.append(columns)
        return rows
    
class JigsawPainting(object):
    def __init__(self, rows=[2,20], cols=[2,20]):
        self.rows = rows
        self.cols = cols
        
    def __call__(self, image):
        image = image.convert('RGB')
        cols = random.randint(self.cols[0], self.cols[1])
        rows = random.randint(self.rows[0], self.rows[1])
        
        alphabet = string.ascii_letters + string.digits
        password = ''.join(secrets.choice(alphabet) for i in range(16))
        image_folder = '/tmp/' + password
        
        os.makedirs(image_folder, exist_ok=True)
        jigsaw = Jigsaw(image, image_folder, cols, rows)
        
        image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        image_files.sort()
        images = [Image.open(os.path.join(image_folder, image_file)) for image_file in image_files]
        width, height = images[0].size
        collage_width = width * cols
        collage_height = height * rows
        collage = Image.new('RGBA', (collage_width, collage_height), color='white')
        for r in range(rows):
            for c in range(cols):
                image_file = '%dx%d.png'%(c,r)
                collage.paste(Image.open(os.path.join(image_folder, image_file)), (c * width, r * height))
        
        white_background = Image.new('RGB', collage.size, (255, 255, 255))
        white_background.paste(collage, mask=collage.split()[3])
        os.system('rm -rf '+ image_folder)
        return white_background.resize((256,256))
    
class FlipRotateCollage(object):
    def __call__(self, image):
        image = image.convert('RGB')
        images = [image.transpose(Image.Transpose.FLIP_LEFT_RIGHT), \
          image.transpose(Image.Transpose.FLIP_TOP_BOTTOM), \
          image.transpose(Image.Transpose.ROTATE_180), \
          image]
        random.shuffle(images)
        width, height = images[0].size
        collage_width = width * 2
        collage_height = height * 2

        collage = Image.new('RGB', (collage_width, collage_height))

        idx = 0
        for r in range(2):
            for c in range(2):
                collage.paste(images[idx], (c * width, r * height))
                idx += 1
        return collage.resize((256,256))

class Elastic(object):
    def __init__(self,alpha=1, sigma=50, alpha_affine=[100, 500], interpolation=(0,4), border_mode=(0,5), value=None, mask_value=None,always_apply=False, p=1):
        self.alpha = alpha
        self.alpha_affine = alpha_affine
        self.sigma = sigma
        self.alpha_affine = alpha_affine
        self.interpolation = interpolation
        self.border_mode = border_mode
        self.value = value
        self.mask_value = mask_value
        self.always_apply = always_apply
        self.p = p
    
    def __call__(self,img):
        img = np.array(img)
        interpolation = random.randint(self.interpolation[0],self.interpolation[1])
        border_mode = random.randint(self.border_mode[0],self.border_mode[1])
        alpha_affine = random.randint(self.alpha_affine[0],self.alpha_affine[1])
        transform = A.ElasticTransform(alpha=self.alpha, sigma=self.sigma, alpha_affine=alpha_affine, interpolation=interpolation, border_mode=border_mode, value=self.value, mask_value=self.mask_value,always_apply=self.always_apply,p=self.p)
        new_img = transform(image=img)['image']
        new_img = Image.fromarray(new_img)
        return new_img
    
class Affine(object):
    def __init__(self,scale=(0,1), translate_percent=(0,1), translate_px=None, shear=(-45,45), interpolation=(0,5), mask_interpolation=(0,1), cval=(0,255), cval_mask=(0,255), mode=(0,5), fit_output=(0,1), keep_ratio=False,always_apply=False, p=1):
        self.scale = scale
        self.translate_percent = translate_percent
        self.translate_px = translate_px
        self.shear = shear
        self.interpolation = interpolation
        self.mask_interpolation = mask_interpolation
        self.mode = mode
        self.cval = cval
        self.cval_mask = cval_mask
        self.fit_output = fit_output
        self.keep_ratio = keep_ratio
        self.always_apply = always_apply
        self.p = p
    
    def __call__(self,img):
        img = img.convert('RGB')
        img = np.array(img)
        scale = random.uniform(self.scale[0],self.scale[1])
        translate_percent = random.uniform(self.translate_percent[0],self.translate_percent[1])
        shear = random.uniform(self.shear[0],self.shear[1])
        interpolation = random.randint(self.interpolation[0],self.interpolation[1])
        mask_interpolation = random.randint(self.mask_interpolation[0],self.mask_interpolation[1])
        cval = random.uniform(self.cval[0],self.cval[1])
        cval_mask = random.uniform(self.cval_mask[0],self.cval_mask[1])
        mode = random.randint(self.mode[0],self.mode[1])
        fit_output = random.randint(self.fit_output[0],self.fit_output[1])
        keep_ratio = self.keep_ratio
        transform = A.Affine(scale= scale, translate_percent=translate_percent, translate_px=self.translate_px, rotate=0, shear=shear, interpolation=interpolation, mask_interpolation=mask_interpolation, cval=cval, cval_mask=cval_mask, mode=mode, fit_output=fit_output, keep_ratio=keep_ratio, always_apply=self.always_apply,p=self.p)
        new_img = transform(image=img)['image']
        new_img = Image.fromarray(new_img)
        return new_img
    
class FancyPCA(object):
    def __init__(self,alpha=3, always_apply=False, p=1):
        self.alpha = alpha
        self.always_apply = always_apply
        self.p = p
    
    def __call__(self,img):
        img = img.convert('RGB')
        img = np.array(img)
        transform = A.FancyPCA(alpha=self.alpha,always_apply=self.always_apply,p=self.p)
        new_img = transform(image=img)['image']
        new_img = Image.fromarray(new_img)
        return new_img
    
class GridDistortion(object):
    '''
    https://github.com/Project-MONAI/MONAI/issues/2186
    '''
    def __init__(self,num_steps=(1,10), distort_limit=(0.5,1), interpolation=(0,4), border_mode=(0,5), value=None, mask_value=None, normalized=False,always_apply=False, p=1):
        self.num_steps = num_steps
        self.distort_limit = distort_limit
        self.interpolation = interpolation
        self.border_mode = border_mode
        self.value = value
        self.mask_value = mask_value
        self.normalized = normalized
        self.always_apply = always_apply
        self.p = p
    
    def __call__(self,img):
        img = img.convert('RGB')
        img = np.array(img)
        num_step = random.randint(self.num_steps[0],self.num_steps[1])
        distort_limit = random.uniform(self.distort_limit[0],self.distort_limit[1])
        interpolation = random.randint(self.interpolation[0],self.interpolation[1])
        border_mode = random.randint(self.border_mode[0],self.border_mode[1])
        transform = A.GridDistortion(num_steps=num_step, distort_limit=distort_limit, interpolation=interpolation, border_mode=border_mode, value=self.value, mask_value=self.mask_value, normalized=self.normalized,always_apply=self.always_apply,p=self.p)
        new_img = transform(image=img)['image']
        new_img = Image.fromarray(new_img)
        return new_img
    

class HistogramMatching(object):
    '''
    https://en.wikipedia.org/wiki/Histogram_matching
    '''
    def __init__(self,reference_images = 'gsdata/VSC/data/emoji/', blend_ratio=(0.5, 1.0), read_fn=lambda x: x,always_apply=False, p=1):
        self.reference_images = reference_images
        self.blend_ratio = blend_ratio
        self.read_fn = read_fn
        self.always_apply = always_apply
        self.p = p
    
    def __call__(self,img):
        img = img.convert('RGB')
        img = np.array(img.convert("RGB"))
        target_img = Image.open(self.reference_images + random.choice(os.listdir(self.reference_images)))
        target_img = target_img.convert('RGB')
        target_img = np.array(target_img)
        transform = A.HistogramMatching(reference_images=[target_img],blend_ratio=self.blend_ratio,read_fn = self.read_fn,always_apply=self.always_apply,p=self.p)
        new_img = transform(image=img)['image']
        new_img = Image.fromarray(new_img)
        return new_img
    
class ISONoise(object):
    '''
    Apply camera sensor noise
    '''
    
    def __init__(self,color_shift=(0.1, 1), intensity=(0.4, 1),always_apply=False, p=1):
        self.color_shift = color_shift
        self.intensity = intensity
        self.always_apply = always_apply
        self.p = p
    
    def __call__(self,img):
        img = img.convert('RGB')
        img = np.array(img)
        transform = A.ISONoise(color_shift=self.color_shift,intensity=self.intensity,always_apply=self.always_apply,p=self.p)
        new_img = transform(image=img)['image']
        new_img = Image.fromarray(new_img)
        return new_img
    
class MultiplicativeNoise(object):
    '''
    https://en.wikipedia.org/wiki/Multiplicative_noise
    '''
    
    def __init__(self,multiplier=(1.5, 3), per_channel=True, elementwise=True,always_apply=False, p=1):
        self.multiplier = multiplier
        self.per_channel = per_channel
        self.elementwise = elementwise
        self.always_apply = always_apply
        self.p = p
    
    def __call__(self,img):
        img = img.convert('RGB')
        img = np.array(img)
        transform = A.MultiplicativeNoise(multiplier=self.multiplier,per_channel=self.per_channel,elementwise=self.elementwise,always_apply=self.always_apply,p=self.p)
        new_img = transform(image=img)['image']
        new_img = Image.fromarray(new_img)
        return new_img
    
class Posterize(object):
    '''
    https://en.wikipedia.org/wiki/Posterization
    '''
    def __init__(self,num_bits=[1,4],always_apply=False, p=1):
        self.num_bits = num_bits
        self.always_apply = always_apply
        self.p = p
    
    def __call__(self,img):
        img = img.convert('RGB')
        num_bit = random.randint(self.num_bits[0],self.num_bits[1])
        img = np.array(img)
        transform = A.Posterize(num_bits=num_bit,always_apply=self.always_apply,p=self.p)
        new_img = transform(image=img)['image']
        new_img = Image.fromarray(new_img)
        return new_img
    
class RandomGamma(object):
    '''
    https://jax.readthedocs.io/en/latest/_autosummary/jax.random.gamma.html
    '''
    def __init__(self,gamma_limit=(200, 1000), eps=None,always_apply=False, p=1):
        self.gamma_limit = gamma_limit
        self.eps = eps
        self.always_apply = always_apply
        self.p = p
    
    def __call__(self,img):
        img = img.convert('RGB')
        img = np.array(img)
        transform = A.RandomGamma(gamma_limit=self.gamma_limit, eps=self.eps, always_apply=self.always_apply,p=self.p)
        new_img = transform(image=img)['image']
        new_img = Image.fromarray(new_img)
        return new_img
    
class RandomShadowy(object):
    def __init__(self, no_of_shadows=[1,10]):
        self.no_of_shadows = no_of_shadows
        
    def __call__(self, image):
        image = image.convert('RGB')
        no_of_shadows = random.randint(self.no_of_shadows[0],self.no_of_shadows[1])
        image= hp.load_image(image)
        shadowy_images= am.add_shadow(image, no_of_shadows=no_of_shadows)
        return Image.fromarray(cv2.cvtColor(shadowy_images, cv2.COLOR_BGR2RGB))
    
class RandomGravel(object):
    def __init__(self, no_of_patches=[10,100]):
        self.no_of_patches = no_of_patches
        
    def __call__(self, image):
        image = image.convert('RGB')
        no_of_patches = random.randint(self.no_of_patches[0],self.no_of_patches[1])
        image= hp.load_image(image)
        bad_road_images= am.add_gravel(image, no_of_patches=no_of_patches)
        return Image.fromarray(cv2.cvtColor(bad_road_images, cv2.COLOR_BGR2RGB))
    
class RandomSunFlare(object):
    def __init__(self, no_of_flare_circles=[5,20]):
        self.no_of_flare_circles = no_of_flare_circles
        
    def __call__(self, image):
        image = image.convert('RGB')
        no_of_flare_circles = random.randint(self.no_of_flare_circles[0],self.no_of_flare_circles[1])
        image= hp.load_image(image)
        flare_images= am.add_sun_flare(image, no_of_flare_circles=no_of_flare_circles)
        return Image.fromarray(cv2.cvtColor(flare_images, cv2.COLOR_BGR2RGB))
    
class RandomSpeedUp(object):
    def __init__(self, speed_coeff=[0.5,1]):
        self.speed_coeff = speed_coeff
        
    def __call__(self, image):
        image = image.convert('RGB')
        speed_coeff = random.uniform(self.speed_coeff[0],self.speed_coeff[1])
        image= hp.load_image(image)
        flare_images= am.add_speed(image, speed_coeff=speed_coeff)
        return Image.fromarray(cv2.cvtColor(flare_images, cv2.COLOR_BGR2RGB))
    
class SeasonChange(object):
    def __init__(self, choices=3):
        self.choices = choices
    
    def change_season_to_autumn(self, image, r_factor, g_factor, b_factor):
        # Split the image into individual color channels
        r, g, b = image.split()

        # Enhance the color channels using the factors
        r = r.point(lambda i: i * r_factor)
        g = g.point(lambda i: i * g_factor)
        b = b.point(lambda i: i * b_factor)

        # Merge the color channels back together
        return Image.merge('RGB', (r, g, b))

    def change_season_to_winter(self, image, brightness_factor, r_factor, g_factor, b_factor):
        # Adjust the brightness
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(brightness_factor)

        # Split the image into individual color channels
        r, g, b = image.split()

        # Enhance the color channels using the factors
        r = r.point(lambda i: i * r_factor)
        g = g.point(lambda i: i * g_factor)
        b = b.point(lambda i: i * b_factor)

        # Merge the color channels back together
        return Image.merge('RGB', (r, g, b))
    
    def change_season_to_spring(self, image, brightness_factor, contrast_factor, r_factor, g_factor, b_factor):
        # Adjust the brightness
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(brightness_factor)

        # Adjust the contrast
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(contrast_factor)

        # Split the image into individual color channels
        r, g, b = image.split()

        # Enhance the color channels using the factors
        r = r.point(lambda i: i * r_factor)
        g = g.point(lambda i: i * g_factor)
        b = b.point(lambda i: i * b_factor)

        # Merge the color channels back together
        return Image.merge('RGB', (r, g, b))
    
    def change_season_to_summer(self, image, brightness_factor, contrast_factor, r_factor, g_factor, b_factor):
        # Adjust the brightness
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(brightness_factor)

        # Adjust the contrast
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(contrast_factor)

        # Split the image into individual color channels
        r, g, b = image.split()

        # Enhance the color channels using the factors
        r = r.point(lambda i: i * r_factor)
        g = g.point(lambda i: i * g_factor)
        b = b.point(lambda i: i * b_factor)

        # Merge the color channels back together
        return Image.merge('RGB', (r, g, b))
    
    def __call__(self, image):
        image = image.convert('RGB')
        choice = random.randint(0, self.choices)
        if choice == 0:
            return self.change_season_to_spring(image, 1.1, 1.1, 1.0, 1.2, 1.0)
        elif choice == 1:
            return self.change_season_to_summer(image, 1.1, 1.1, 1.2, 1.1, 0.9)
        elif choice == 2:
            return self.change_season_to_autumn(image, 1.2, 0.8, 1.0)
        elif choice == 3:
            return self.change_season_to_winter(image, 1.1, 0.9, 0.9, 1.2)
        
class CorrectExposure(object):     
    def __call__(self, image):
        image = image.convert('RGB')
        image= hp.load_image(image)
        correct_images= am.correct_exposure(image)
        return Image.fromarray(cv2.cvtColor(correct_images, cv2.COLOR_BGR2RGB))
    
class RandomMosaic(object):
    def __init__(self, max_block_size=[10,50]):
        self.max_block_size = max_block_size
        
    def __call__(self, image):
        image = image.convert('RGB')
        max_block_size = random.randint(self.max_block_size[0], self.max_block_size[1])
        
        width, height = image.size
        mosaic_image = Image.new("RGB", (width, height))

        y = 0
        while y < height:
            x = 0
            while x < width:
                # Determine the random size of the block within the maximum limit and image bounds
                block_width = random.randint(1, max_block_size)
                block_height = random.randint(1, max_block_size)
                block_width = min(block_width, width - x)
                block_height = min(block_height, height - y)

                # Create a rectangular patch from the original image
                patch = image.crop((x, y, x + block_width, y + block_height))

                # Calculate the mean color of the patch
                patch_np = np.array(patch)
                mean_color = tuple(np.mean(patch_np.reshape(-1, 3), axis=0).astype(int))

                # Create a solid color image with the mean color and the size of the patch
                mean_patch = Image.new("RGB", (block_width, block_height), mean_color)

                # Paste the mean color patch back into the new image
                mosaic_image.paste(mean_patch, (x, y))

                x += block_width
            y += block_height

        return mosaic_image

class MultiPartsCollage(object):
    def __init__(self, repeat_horizontal = 5, repeat_vertical = 5):
        self.repeat_horizontal = repeat_horizontal
        self.repeat_vertical = repeat_vertical
    
    def __call__(self, image):
        image = image.convert('RGB')
        repeat_horizontal = random.randint(2,self.repeat_horizontal)
        repeat_vertical = random.randint(2,self.repeat_vertical)
        
        image = image.resize((256,256))
        width, height = image.size
        new_width = width * repeat_horizontal
        new_height = height * repeat_vertical
        new_image = Image.new('RGB', (new_width, new_height))

        for i in range(repeat_horizontal):
            for j in range(repeat_vertical):
                new_image.paste(transforms.RandomResizedCrop(256, scale=(0.02,1))(image), (i * width, j * height))

        return new_image.resize((256,256))

class RandomCollage(object):
    def __init__(self, path = 'gsdata/VSC/data/training_images_9/', N = [2,5], M = [2,5]):
        self.path = path
        self.images = os.listdir(path)
        self.N = N
        self.M = M
        
    def __call__(self, ori_image):
        ori_image = ori_image.convert('RGB')
        img_width = 256
        img_height = 256
        N = random.randint(self.N[0], self.N[1])
        M = random.randint(self.M[0], self.M[1])
        collage_width = N * img_width
        collage_height = M * img_height
        collage = Image.new("RGB", (collage_width, collage_height))
        image_paths = [Image.open(self.path + i) for i in random.sample(self.images, N*M-1)] + [ori_image]
        random.shuffle(image_paths)
        for i, image_path in enumerate(image_paths):
            row = i // N
            col = i % N
            x = col * img_width
            y = row * img_height
            image = image_path.resize((img_width, img_height))
            collage.paste(image, (x, y))
        return collage.resize((256,256))
    
class Spatter(object):
    def __init__(self,mean=0.65, std=0.3, gauss_sigma=2, cutout_threshold=0.68, intensity=0.6, mode='rain', color=None,always_apply=False, p=1):
        self.mean = mean
        self.std = std
        self.gauss_sigma = gauss_sigma
        self.cutout_threshold = cutout_threshold
        self.intensity = intensity
        self.mode = mode
        self.color = color
        self.always_apply = always_apply
        self.p = p
    
    def __call__(self,img):
        img = np.array(img.convert('RGB'))
        transform = A.Spatter(mean=self.mean,std=self.std,gauss_sigma=self.gauss_sigma,cutout_threshold=self.cutout_threshold,intensity=self.intensity,mode=self.mode,always_apply=self.always_apply,p=self.p)
        new_img = transform(image=img)['image']
        new_img = Image.fromarray(new_img)
        return new_img
    
class Superpixels(object):
    '''
    https://www.tu-chemnitz.de/etit/proaut/en/research/superpixel.html
    '''
    def __init__(self,p_replace=0.1, n_segments=200, max_size=128, interpolation=1,always_apply=False, p=1):
        self.p_replace = p_replace
        self.n_segments = n_segments
        self.max_size = max_size
        self.interpolation = interpolation
        self.always_apply = always_apply
        self.p = p
    
    def __call__(self,img):
        img = img.convert('RGB')
        img = np.array(img)
        transform = A.Superpixels(p_replace=self.p_replace,n_segments=self.n_segments,max_size=self.max_size,interpolation=self.interpolation,always_apply=self.always_apply,p=self.p)
        new_img = transform(image=img)['image']
        new_img = Image.fromarray(new_img)
        return new_img
    
class RandomHeatmap(object):
    def __init__(self, choices = 21):
        self.choices = choices
    
    def __call__(self, image):
        image = image.convert('RGB')
        choice = random.randint(0, self.choices)
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        heatmap = cv2.applyColorMap((image), choice)
        img_pil = Image.fromarray(cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)) 
        return img_pil
    
class PolarWarp(object):
    def __call__(self, image):
        image = image.convert('RGB')
        aug = iaa.WithPolarWarping(iaa.CropAndPad(percent=(-0.4, 0.4)))
        return Image.fromarray(aug(images = [np.array(image)])[0])

class DropChannel(object):
    def __call__(self, image):
        image = image.convert('RGB')
        aug = iaa.Dropout2d(p=1)
        return Image.fromarray(aug(images = [np.array(image)])[0])
    
class ColorQuantization(object):
    def __init__(self, choices=2):
        self.choices = choices
    
    def __call__(self, image):
        image = image.convert('RGB')
        choice = random.randint(0, self.choices)
        if choice == 0:
            aug = iaa.KMeansColorQuantization(n_colors=(2,10))
        elif choice == 1:
            aug = iaa.UniformColorQuantization(n_colors=(2,10))
        else:
            aug = iaa.UniformColorQuantizationToNBits(nb_bits=(2, 6))
        return Image.fromarray(aug(images = [np.array(image)])[0])

class Emboss(object):
    def __call__(self, image):
        image = image.convert('RGB')
        aug = iaa.pillike.FilterEmboss()
        return Image.fromarray(aug(images = [np.array(image)])[0])
    
class Voronoi(object):
    '''
    https://en.wikipedia.org/wiki/Voronoi_diagram
    '''
    def __init__(self, choices=2):
        self.choices = choices
    def __call__(self, image):
        image = image.convert('RGB')
        choice = random.randint(0, self.choices)
        if choice == 0:
            aug = iaa.UniformVoronoi((200,800), p_replace=1)
        elif choice == 1:
            aug = iaa.RegularGridVoronoi(p_replace=1)
        elif choice == 2:
            aug = iaa.RelativeRegularGridVoronoi((0.01,0.2), (0.01,0.2), p_replace=1)
        return Image.fromarray(aug(images = [np.array(image)])[0])

class RandomColorSpace(object):
    def __init__(self, choices=6):
        self.choices = choices
        
    def __call__(self, image):
        image = image.convert('RGB')
        choice = random.randint(0, self.choices)
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        if choice == 0:
            hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            return Image.fromarray(hsv_image)
        
        elif choice == 1:
            hls_image = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
            return Image.fromarray(hls_image)
        
        elif choice == 2:
            lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            return Image.fromarray(lab_image) 
        
        elif choice == 3:
            YCrCb_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
            return Image.fromarray(YCrCb_image)
        
        elif choice == 4:
            luv_image = cv2.cvtColor(image, cv2.COLOR_BGR2LUV) 
            return Image.fromarray(luv_image)
        
        elif choice == 5:
            xyz_image = cv2.cvtColor(image, cv2.COLOR_BGR2XYZ) 
            return Image.fromarray(xyz_image)
        
        elif choice == 6:
            yuv_image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV) 
            return Image.fromarray(yuv_image)
        
class OldSchool(object):
    def __init__(self, choices=1, scale_factors = [0.1, 0.6], \
                scanline_intensity=[50,1000], scanline_spacing=[2,15], barrel_distortion_k1=[0.1,0.3], \
                barrel_distortion_k2=[0.1,0.3], vignette_intensity = [0.3, 1]):
        self.choices = choices
        self.scale_factors = scale_factors
        self.scanline_intensity = scanline_intensity
        self.scanline_spacing = scanline_spacing
        self.barrel_distortion_k1 = barrel_distortion_k1
        self.barrel_distortion_k2 = barrel_distortion_k2
        self.vignette_intensity = vignette_intensity
        
    def cga_palette(self, image):
        palette = np.array([
            [0, 0, 0],      # Black
            [0, 170, 0],    # Green
            [170, 0, 0],    # Red
            [170, 170, 170] # Gray
        ], dtype=np.uint8)

        # Quantize the image to the CGA palette using k-means clustering
        reshaped_image = image.reshape(-1, 3)
        _, labels, _ = cv2.kmeans(reshaped_image.astype(np.float32), 4, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1.0), 10, cv2.KMEANS_RANDOM_CENTERS)
        quantized_image = palette[labels.flatten()].reshape(image.shape)

        return quantized_image
    
    def add_scanlines(self, image, intensity, line_spacing):
        overlay = np.zeros_like(image, dtype=np.float32)

        for i in range(0, image.shape[0], line_spacing):
            overlay[i:i+1, :, :] = intensity

        return np.clip(image.astype(np.float32) - overlay, 0, 255).astype(np.uint8)

    def apply_barrel_distortion(self, image, k1, k2):
        height, width = image.shape[:2]
        fx, fy = width / 2, height / 2
        camera_matrix = np.array([[fx, 0, width/2], [0, fy, height/2], [0, 0, 1]])
        distortion_coeffs = np.array([k1, k2, 0, 0])

        map1, map2 = cv2.initUndistortRectifyMap(camera_matrix, distortion_coeffs, None, camera_matrix, (width, height), cv2.CV_32FC1)
        return cv2.remap(image, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    def apply_vignette(self, image, intensity):
        height, width = image.shape[:2]
        mask = np.zeros((height, width), dtype=np.float32)

        x, y = np.meshgrid(np.linspace(0, width - 1, width), np.linspace(0, height - 1, height))
        x = (x - width / 2) / (width / 2)
        y = (y - height / 2) / (height / 2)
        r = np.sqrt(x ** 2 + y ** 2)

        mask = 1 - r * intensity
        mask = np.clip(mask, 0, 1)

        return np.clip(image.astype(np.float32) * mask[:, :, np.newaxis], 0, 255).astype(np.uint8)
        
    def __call__(self, image):
        image = image.convert('RGB')
        choice = random.randint(0, self.choices)
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        if choice == 0:
            cga_image = self.cga_palette(image)
            scale_factor = random.uniform(self.scale_factors[0], self.scale_factors[1])
            low_res_image = cv2.resize(cga_image, (0, 0), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_NEAREST)
            upscaled_image = cv2.resize(low_res_image, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
            return Image.fromarray(cv2.cvtColor(upscaled_image, cv2.COLOR_BGR2RGB))
        
        elif choice == 1:
            # Apply CRT-style effects
            scanline_intensity = random.randint(self.scanline_intensity[0], self.scanline_intensity[1])
            scanline_spacing = random.randint(self.scanline_spacing[0], self.scanline_spacing[1])
            barrel_distortion_k1 = random.uniform(self.barrel_distortion_k1[0], self.barrel_distortion_k1[1])
            barrel_distortion_k2 = random.uniform(self.barrel_distortion_k2[0], self.barrel_distortion_k2[1])
            vignette_intensity = random.uniform(self.vignette_intensity[0], self.vignette_intensity[1])
            scanline_image = self.add_scanlines(image, scanline_intensity, scanline_spacing)
            distorted_image = self.apply_barrel_distortion(scanline_image, barrel_distortion_k1, barrel_distortion_k2)
            crt_style_image = self.apply_vignette(distorted_image, vignette_intensity)

            return Image.fromarray(cv2.cvtColor(crt_style_image, cv2.COLOR_BGR2RGB))

class Ascii(object):
    def __init__(self, font_path = 'gsdata/VSC/data/fonts/', font_size = [20,80], colored = [0, 1]):
        self.fonts = os.listdir(font_path)
        self.font_path = font_path
        self.font_size = font_size
        self.colored = colored
    
    def image_to_ascii(self, image, output_width, output_height, font_path, font_size, colored):#73
        # Convert the image to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Resize the image
        resized_image = cv2.resize(gray_image, (output_width, output_height), interpolation=cv2.INTER_AREA)

        # Normalize the pixel intensities to the range [0, 255]
        normalized_image = cv2.normalize(resized_image, None, 0, 255, cv2.NORM_MINMAX)

        # Create the ASCII character set
        ascii_chars = '@%#*+=-:. '

        # Map the image's pixel intensities to ASCII characters
        ascii_image = np.zeros_like(resized_image, dtype=np.dtype('U1'))
        for i in range(len(ascii_chars)):
            ascii_image[(normalized_image >= i * 255 // len(ascii_chars)) & (normalized_image <= (i + 1) * 255 // len(ascii_chars))] = ascii_chars[i]

        # Create the colored ASCII text mosaic using the original image's colors
        colored_image = cv2.resize(image, (output_width, output_height), interpolation=cv2.INTER_AREA) if colored else None
        output_image = Image.new('RGB', (output_width * font_size, output_height * font_size), color=(255, 255, 255))
        draw = ImageDraw.Draw(output_image)
        font = ImageFont.truetype(font_path, font_size)

        for i in range(output_height):
            for j in range(output_width):
                if colored:
                    draw.text((j * font_size, i * font_size), ascii_image[i, j], font=font, fill=tuple(colored_image[i, j]))
                else:
                    draw.text((j * font_size, i * font_size), ascii_image[i, j], font=font, fill=(0, 0, 0))

        return output_image
    
    def __call__(self, image):
        image = image.convert('RGB')
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        output_width, output_height = 200, 100
        font_path = self.font_path + '/' + random.choice(self.fonts)
        font_size = random.randint(self.font_size[0], self.font_size[1])
        colored = random.randint(self.colored[0], self.colored[1])
        ascii_image = self.image_to_ascii(image, output_width, output_height, font_path, font_size, colored)
        return ascii_image.resize((256,256))
    
class PixelMelt(object):
    def __init__(self, direction=[0,1], amount=[0,1], threshold=[80,180]):
        self.direction = direction
        self.amount = amount
        self.threshold = threshold
    
    def pixel_melt(self, image, direction='vertical', amount=0.5, threshold=128):
        h, w = image.shape[:2]
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        sorted_image = np.copy(image)

        if direction == 'vertical':
            for x in range(w):
                column = image[:, x]
                gray_column = gray_image[:, x]
                mask = gray_column > threshold
                if np.any(mask):
                    sorted_column = column[mask]
                    sorted_column = sorted(sorted_column, key=lambda pixel: np.sum(pixel), reverse=True)
                    sorted_image[mask, x] = sorted_column[:int(len(sorted_column) * amount)] + sorted_column[int(len(sorted_column) * amount):][::-1]
        elif direction == 'horizontal':
            for y in range(h):
                row = image[y]
                gray_row = gray_image[y]
                mask = gray_row > threshold
                if np.any(mask):
                    sorted_row = row[mask]
                    sorted_row = sorted(sorted_row, key=lambda pixel: np.sum(pixel), reverse=True)
                    sorted_image[y, mask] = sorted_row[:int(len(sorted_row) * amount)] + sorted_row[int(len(sorted_row) * amount):][::-1]

        return sorted_image

        
    def __call__(self, image):
        image = cv2.cvtColor(np.array(image.convert('RGB')), cv2.COLOR_RGB2BGR)
        dire = random.randint(self.direction[0], self.direction[1])
        if dire == 0:
            direction = 'horizontal'
        else:
            direction = 'vertical'
        amount = random.uniform(self.amount[0], self.amount[1])
        threshold = random.randint(self.threshold[0], self.threshold[1])
        melted_image = self.pixel_melt(image, direction, amount, threshold)
        return Image.fromarray(cv2.cvtColor(melted_image, cv2.COLOR_BGR2RGB))
    
class RandomStyle(object):
    def __init__(self, image_path = 'gsdata/VSC/data/training_images_9/'):
        self.images = os.listdir(image_path)
        self.image_path = image_path
    def __call__(self, image):
        image = image.convert('RGB')
        alphabet = string.ascii_letters + string.digits
        password = ''.join(secrets.choice(alphabet) for i in range(17))
        image.save('/tmp/%s.jpg'%password, quality=100)
        style_path = self.image_path + random.choice(self.images)
        os.system('python gsdata/AnyPattern/Generate/PyTorch-Multi-Style-Transfer/experiments/main.py eval \
                --content-image /tmp/%s.jpg \
                --style-image %s \
                --model gsdata/AnyPattern/Generate/PyTorch-Multi-Style-Transfer/experiments/models/21styles.model \
                --content-size 1024 --cuda 0 --output-image /tmp/%s.jpg'%(password, style_path, password)
        )
        style_image = Image.open('/tmp/%s.jpg'%password)
        os.remove('/tmp/%s.jpg'%password)
        return style_image

class AnimeGAN(object):
    def __init__(self, choices = 1):
        self.choices = choices
    def __call__(self, image):
        image = image.convert('RGB')
        choice = random.randint(0, self.choices)
        alphabet = string.ascii_letters + string.digits
        password = ''.join(secrets.choice(alphabet) for i in range(18))
        image.save('/tmp/%s.jpg'%password, quality=100)
        src = '/tmp/%s.jpg'%password
        dst = '/tmp/%s.png'%password
        if choice == 0:
            os.system('python ./pytorch-animeGAN/inference_image.py \
                    --checkpoint ./pytorch-animeGAN/.cache/generator_hayao.pth \
                    --src %s --dest %s'%(src,dst))
        else:
            os.system('python ./pytorch-animeGAN/inference_image.py \
                    --checkpoint ./pytorch-animeGAN/.cache/generator_shinkai.pth \
                    --src %s --dest %s'%(src,dst))
        ani_image = Image.open('/tmp/%s.png'%password).convert('RGB')
        os.remove('/tmp/%s.jpg'%password)
        os.remove('/tmp/%s.png'%password)
        return ani_image

class AdversarialAttack(object):
    def __init__(self):
        self.classification_models = torchvision.models.list_models(module=torchvision.models)
        self.classification_models.remove('convnext_large')
        self.classification_models.remove('vit_h_14')
        self.classification_models.remove('resnet152')
        self.classification_models.remove('efficientnet_b7')
    
    def __call__(self, image):
        image = image.convert('RGB')
        cls = random.choice(self.classification_models)
        print(cls)
        model = torchvision.models.get_model(cls, weights="DEFAULT")
        transform_pipeline = transforms.Compose([
            transforms.ToTensor(),
        ])
        
        if ('vit' in cls) or ('swin' in cls):
            image = image.resize((224,224))
        
        try:
            imgg = transform_pipeline(image).unsqueeze(0)
            imgg = imgg.double()
            model = model.double()
            atk = torchattacks.PGD(model, eps=8/255, alpha=2/255, steps=4)
            adv_images = atk(imgg, torch.Tensor(1).long())
            new_image = transforms.ToPILImage()(adv_images[0])
        except:
            image = image.resize((224,224))
            imgg = transform_pipeline(image).unsqueeze(0)
            imgg = imgg.double()
            model = model.double()
            atk = torchattacks.PGD(model, eps=8/255, alpha=2/255, steps=4)
            adv_images = atk(imgg, torch.Tensor(1).long())
            new_image = transforms.ToPILImage()(adv_images[0])
            new_image = new_image.resize((256,256))
        
        return new_image.resize((256,256))
    
class CLAHE(object):
    '''
    https://en.wikipedia.org/wiki/Adaptive_histogram_equalization
    '''
    def __init__(self,clip_limit=[4.0, 8.0], tile_grid_size_h=[8,40], tile_grid_size_w=[8,40], always_apply=False, p=1):
        self.clip_limit = clip_limit
        self.tile_grid_size_h = tile_grid_size_h
        self.tile_grid_size_w = tile_grid_size_w
        self.always_apply = always_apply
        self.p = p
    
    def __call__(self,img):
        img = img.convert('RGB')
        img = np.array(img)
        tile_grid_size_h = random.randint(self.tile_grid_size_h[0], self.tile_grid_size_h[1])
        tile_grid_size_w = random.randint(self.tile_grid_size_w[0], self.tile_grid_size_w[1])
        transform = A.CLAHE(clip_limit=self.clip_limit,\
                            tile_grid_size=(tile_grid_size_h,tile_grid_size_w),always_apply=self.always_apply,p=self.p)
        new_img = transform(image=img)['image']
        new_img = Image.fromarray(new_img)
        return new_img
    
class OverlayScreenshot(object):
    def __init__(self, choice = 1):
        self.choice = choice
        
    def __call__(self, image):
        image = image.convert('RGB')
        choice = random.randint(0,self.choice)
        if choice == 0:
            path = 'anaconda3/envs/torch2/lib/python3.9/site-packages/augly/assets/screenshot_templates/mobile.png'
        else:
            path = 'anaconda3/envs/torch2/lib/python3.9/site-packages/augly/assets/screenshot_templates/web.png'
            
        return OverlayOntoScreenshot(template_filepath=path)(image).resize((256,256))
    
class DisLocation(object):
    
    def __init__(self, choice = 1, n_cuts = [1, 10]):
        self.choice = choice
        self.n_cuts = n_cuts
        
    def random_split_image_hori(self, img, n_cuts):
        width, height = img.size
        cut_points = sorted(random.sample(range(1, height - 1), n_cuts))
        cut_points = [0] + cut_points + [height]
        result = Image.new('RGB', (width, height))

        y_offset = 0
        for i in range(len(cut_points) - 1):
            y1, y2 = cut_points[i], cut_points[i + 1]
            region_height = y2 - y1

            region = img.crop((0, y1, width, y2))

            cut_x = random.randint(1, width - 1)

            left_part = region.crop((0, 0, cut_x, region_height))
            right_part = region.crop((cut_x, 0, width, region_height))

            new_region = Image.new('RGB', (width, region_height))
            new_region.paste(right_part, (0, 0))
            new_region.paste(left_part, (width-cut_x, 0))

            result.paste(new_region, (0, y_offset))
            y_offset += region_height
        return result
    
    def random_split_image_veri(self, img, n_cuts):
        width, height = img.size
        cut_points = sorted(random.sample(range(1, width - 1), n_cuts))
        cut_points = [0] + cut_points + [width]
        result = Image.new('RGB', (width, height))
        x_offset = 0
        for i in range(len(cut_points) - 1):
            x1, x2 = cut_points[i], cut_points[i + 1]
            region_width = x2 - x1
            region = img.crop((x1, 0, x2, height))
            cut_y = random.randint(1, height - 1)
            top_part = region.crop((0, 0, region_width, cut_y))
            bottom_part = region.crop((0, cut_y, region_width, height))

            new_region = Image.new('RGB', (region_width, height))
            new_region.paste(bottom_part, (0, 0))
            new_region.paste(top_part, (0, height - cut_y))
            result.paste(new_region, (x_offset, 0))
            x_offset += region_width

        return result
    
    def __call__(self, image):
        image = image.convert('RGB')
        choice = random.randint(0, self.choice)
        n_cuts = random.randint(self.n_cuts[0], self.n_cuts[1])
        if choice == 0:
            return self.random_split_image_hori(image, n_cuts)
        else:
            return self.random_split_image_veri(image, n_cuts)

class DropArea(object):
    def __init__(self, choice = 1, drop_prob = [0.2, 0.7]):
        self.choice = choice
        self.drop_prob = drop_prob
    
    def __call__(self, image):
        image = image.convert('RGB')
        choice = random.randint(0, self.choice)
        drop_prob = random.uniform(self.drop_prob[0], self.drop_prob[1])
        
        t = transforms.Compose([
            transforms.ToTensor(),
        ])
        img_tensor = t(image).unsqueeze(0)
        
        if choice == 0:
            sp = img_tensor.shape
            block_size = int(random.uniform((sp[2] + sp[3])/2*0.2, (sp[2] + sp[3])/2*0.5))
            drop_block = DropBlock2D(block_size=block_size, drop_prob=drop_prob)
        else:
            drop_block = DropBlock2D(block_size=1, drop_prob=drop_prob)
        
        img_tensor_drop_block = drop_block(img_tensor)
        return transforms.ToPILImage()(img_tensor_drop_block[0])

class Waveblock(object):
    '''
    https://ieeexplore.ieee.org/abstract/document/9677903
    '''
    def __init__(self, choice = 1, ratio = [0.2, 0.8], enlarge = [1.5, 4]):
        self.choice = choice
        self.ratio = ratio
        self.enlarge = enlarge
        
    def __call__(self, image):
        image = image.convert('RGB')
        choice = random.randint(0, self.choice)
        ratio = random.uniform(self.ratio[0], self.ratio[1])
        enlarge = random.uniform(self.enlarge[0], self.enlarge[1])
        
        t = transforms.Compose([
            transforms.ToTensor(),
        ])
        x = t(image).unsqueeze(0)
        h, w = x.size()[-2:]
        if choice == 0:
            rh = round(ratio * h)
            sx = random.randint(0, h-rh)
            mask = (x.new_ones(x.size()))*enlarge
            mask[:, :, sx:sx+rh, :] = 1
        else:
            rw = round(ratio * w)
            sy = random.randint(0, w-rw)
            mask = (x.new_ones(x.size()))*enlarge
            mask[:,:,:,sy:sy+rw] = 1
        x = x * mask
        return transforms.ToPILImage()(x[0])

class RandomShape(object):
    def __init__(self, crop_h = [80, 200], crop_w = [80,200], choices = [0,9], \
                 ellipse_ratio_h = [0.4, 1.0], ellipse_ratio_w = [0.4, 1.0]):
        self.crop_h = crop_h
        self.crop_w = crop_w
        self.choices = choices
        self.ellipse_ratio_h = ellipse_ratio_h
        self.ellipse_ratio_w = ellipse_ratio_w
        
    def create_polygon_mask(self, size, sides):
        mask = Image.new('L', size, 0)
        draw = ImageDraw.Draw(mask)
        width, height = size

        vertices = [
            (width * (0.5 + 0.5 * math.cos(2 * math.pi * i / sides)),
             height * (0.5 + 0.5 * math.sin(2 * math.pi * i / sides)))
            for i in range(sides)
        ]

        draw.polygon(vertices, fill=255)
        return mask
    
    def create_circle_mask(self, size):
        mask = Image.new('L', size, 0)
        draw = ImageDraw.Draw(mask)
        width, height = size

        draw.ellipse([0, 0, width, height], fill=255)
        return mask

    def create_ellipse_mask(self, size, ellipse_ratio=(0.5, 0.75)):
        mask = Image.new('L', size, 0)
        draw = ImageDraw.Draw(mask)
        width, height = size

        ellipse_width = int(width * ellipse_ratio[0])
        ellipse_height = int(height * ellipse_ratio[1])

        draw.ellipse([(width - ellipse_width) // 2, (height - ellipse_height) // 2, 
                      (width + ellipse_width) // 2, (height + ellipse_height) // 2], fill=255)
        return mask


    def create_heart_mask(self, size):
        mask = Image.new('L', size, 0)
        draw = ImageDraw.Draw(mask)
        width, height = size

        half_width = width // 2
        draw.ellipse([0, 0, half_width, half_width], fill=255)
        draw.ellipse([half_width, 0, width, half_width], fill=255)
        draw.polygon([(0, half_width // 2), (width, half_width // 2), (width // 2, height)], fill=255)
        return mask

    def create_star_mask(self, size):
        mask = Image.new('L', size, 0)
        draw = ImageDraw.Draw(mask)
        width, height = size

        vertices = [
            (width * 0.5, height * 0.0),
            (width * 0.63, height * 0.37),
            (width * 1.0, height * 0.38),
            (width * 0.69, height * 0.61),
            (width * 0.82, height * 1.0),
            (width * 0.5, height * 0.75),
            (width * 0.18, height * 1.0),
            (width * 0.31, height * 0.61),
            (width * 0.0, height * 0.38),
            (width * 0.37, height * 0.37)
        ]

        draw.polygon(vertices, fill=255)
        return mask

    def crop_shape_from_image(self, image, size):
        width, height = image.size
        crop_width, crop_height = size

        x = random.randint(0, width - crop_width)
        y = random.randint(0, height - crop_height)

        cropped_image = image.crop((x, y, x + crop_width, y + crop_height))
        return cropped_image

    def paste_shape_on_transparent_background(self, shape_image, mask, size):
        background = Image.new('RGBA', size, (0, 0, 0, 0))
        background.paste(shape_image, (0, 0), mask)
        return background
    
    def __call__(self, image):
        image = image.convert('RGB')
        source_image = image.resize((256,256))
        crop_size = (random.randint(self.crop_h[0], self.crop_h[1]), \
                     random.randint(self.crop_w[0], self.crop_w[1]))
        choice = random.randint(self.choices[0], self.choices[1])
        
        ellipse_ratio_h = random.uniform(self.ellipse_ratio_h[0], self.ellipse_ratio_h[1])
        ellipse_ratio_w = random.uniform(self.ellipse_ratio_w[0], self.ellipse_ratio_w[1])
        

        shape = "nonagon"
        
        if choice == 1:
            shape_mask = self.create_circle_mask(crop_size)
        elif choice == 2:
            shape_mask = self.create_ellipse_mask(crop_size, ellipse_ratio=(ellipse_ratio_h, ellipse_ratio_w))
        elif choice == 0:
            shape_mask = self.create_heart_mask(crop_size)
        elif choice == 3:
            shape_mask = self.create_polygon_mask(crop_size, choice)
        elif choice == 4:
            shape_mask = self.create_star_mask(crop_size)
        else:
            shape_mask = self.create_polygon_mask(crop_size, choice)

        cropped_image = self.crop_shape_from_image(source_image, crop_size)
        shape_image = self.paste_shape_on_transparent_background(cropped_image, \
                                                            shape_mask, crop_size).convert('RGB').resize((256,256))
        return shape_image
    
class VanGoghize(object):
    def __init__(self, image_path = 'gsdata/AnyPattern/data/vangogh'):
        self.images = []
        for root, dirs, files in os.walk(image_path):
            for file in files:
                self.images.append(os.path.join(root, file))
        self.images = [i for i in self.images if '._' not in i]
    def __call__(self, image):
        image = image.convert('RGB')
        alphabet = string.ascii_letters + string.digits
        password = ''.join(secrets.choice(alphabet) for i in range(19))
        image.save('/tmp/%s.jpg'%password, quality=100)
        style_path = random.choice(self.images)
        os.system("python gsdata/AnyPattern/Generate/PyTorch-Multi-Style-Transfer/experiments/main.py eval \
                --content-image /tmp/%s.jpg \
                --style-image '%s' \
                --model gsdata/AnyPattern/Generate/PyTorch-Multi-Style-Transfer/experiments/models/21styles.model \
                --content-size 1024 --cuda 0 --output-image /tmp/%s.jpg"%(password, style_path, password)
        )
        style_image = Image.open('/tmp/%s.jpg'%password)
        os.remove('/tmp/%s.jpg'%password)
        return style_image

class MoirePattern(object):
    def __init__(self, pattern_frequency = [(1, 10), (40, 50), (200, 300)]):
        self.pattern_frequency = pattern_frequency
    
    def create_moire_pattern(self, size, frequency):
        x, y = np.meshgrid(np.linspace(0, 1, size[0]), np.linspace(0, 1, size[1]))
        pattern = 0.5 * (1 + np.sin(2 * np.pi * frequency * (x + y)))
        pattern_rgb = np.stack([pattern, pattern, pattern], axis=-1)
        return (255 * pattern_rgb).astype(np.uint8)

    def add_moire_pattern(self, image, pattern_frequency):
        # Load the image

        # Create the Moire pattern
        pattern = self.create_moire_pattern(image.size, pattern_frequency)
        pattern_image = Image.fromarray(pattern)

        # Add the pattern to the image

        image_with_pattern = Image.blend(image, pattern_image, alpha=0.5)

        # Save the new image
        return image_with_pattern
    
    def __call__(self, image):
        image = image.convert('RGB')
        selected_range = random.choice(self.pattern_frequency)
        pattern_frequency = random.randint(selected_range[0], selected_range[1])

        return self.add_moire_pattern(image, pattern_frequency)
    
class Wavelet(object):
    def __init__(self, choice=2):
        self.choice = choice
        
    def apply_wavelet_channel_high(self, channel):
        coeffs = pywt.dwt2(channel, 'haar')
        LL, (LH, HL, HH) = coeffs

        high_frequency = np.zeros_like(channel)
        high_frequency[:LH.shape[0], :LH.shape[1]] = LH
        high_frequency[:HL.shape[0], HL.shape[1]:] = HL
        high_frequency[LH.shape[0]:, :HH.shape[1]] = HH

        return high_frequency
    
    def apply_wavelet_channel_mid(self, channel):
        coeffs = pywt.dwt2(channel, 'haar')
        LL, (LH, HL, HH) = coeffs

        mid_frequency = np.zeros_like(channel)
        mid_frequency[:LL.shape[0], :LL.shape[1]] = LL
        mid_frequency[:LH.shape[0], LH.shape[1]:] = (LH + HL) / 2

        return mid_frequency
    
    def __call__(self, image):
        image = image.convert('RGB')
        image = image.resize((1024, 680))
        choice = random.randint(0, self.choice)
        if choice == 0:
            h, w = image.size
            image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            blue_channel, green_channel, red_channel = cv2.split(image)
            blue_high_frequency = self.apply_wavelet_channel_high(blue_channel)
            green_high_frequency = self.apply_wavelet_channel_high(green_channel)
            red_high_frequency = self.apply_wavelet_channel_high(red_channel)
            high_frequency_image = cv2.merge((blue_high_frequency, green_high_frequency, red_high_frequency))
            ans = Image.fromarray(high_frequency_image[:high_frequency_image.shape[0]//2,:high_frequency_image.shape[1]//2,:])
            return ans.resize((h, w))
        
        elif choice == 1:
            h, w = image.size
            image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            blue_channel, green_channel, red_channel = cv2.split(image)
            blue_mid_frequency = self.apply_wavelet_channel_mid(blue_channel)
            green_mid_frequency = self.apply_wavelet_channel_mid(green_channel)
            red_mid_frequency = self.apply_wavelet_channel_mid(red_channel)
            mid_frequency_image = cv2.merge((blue_mid_frequency, green_mid_frequency, red_mid_frequency))
            ans = Image.fromarray(mid_frequency_image[:mid_frequency_image.shape[0]//2,:mid_frequency_image.shape[1]//2,:])
            return ans.resize((h, w))
        
        else:
            image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            image_ycbcr = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

            coeffs_y = pywt.dwt2(image_ycbcr[:, :, 0], 'haar')
            coeffs_cb = pywt.dwt2(image_ycbcr[:, :, 1], 'haar')
            coeffs_cr = pywt.dwt2(image_ycbcr[:, :, 2], 'haar')

            low_freq_y = coeffs_y[0]
            low_freq_cb = coeffs_cb[0]
            low_freq_cr = coeffs_cr[0]

            low_freq_image = np.stack((low_freq_y, low_freq_cb, low_freq_cr), axis=-1)

            reconstructed_y = pywt.idwt2((low_freq_y, coeffs_y[1]), 'haar')
            reconstructed_cb = pywt.idwt2((low_freq_cb, coeffs_cb[1]), 'haar')
            reconstructed_cr = pywt.idwt2((low_freq_cr, coeffs_cr[1]), 'haar')

            reconstructed_image = np.stack((reconstructed_y, reconstructed_cb, reconstructed_cr), axis=-1)
            reconstructed_image_uint8 = np.clip(reconstructed_image, 0, 255).astype(np.uint8)

            reconstructed_image_bgr = cv2.cvtColor(reconstructed_image_uint8, cv2.COLOR_YCrCb2BGR)
            return Image.fromarray(reconstructed_image_bgr)
        
class ToSepia(object):
    '''
    https://en.wikipedia.org/wiki/Sepia_(color)
    '''
    def __init__(self,always_apply=False, p=1):
        self.always_apply = always_apply
        self.p = p
    
    def __call__(self,img):
        img = img.convert("RGB")
        img = np.array(img)
        transform = A.ToSepia(always_apply=self.always_apply,p=self.p)
        new_img = transform(image=img)['image']
        new_img = Image.fromarray(new_img)
        return new_img

class RandomToneCurve(object):
    '''
    Randomly change the relationship between bright and dark areas of the image by manipulating its tone curve.
    '''
    def __init__(self,scale=[0,1] ,always_apply=False, p=1):
        self.scale = scale
        self.always_apply = always_apply
        self.p = p
    
    def __call__(self,img):
        img = img.convert('RGB')
        scale = random.uniform(self.scale[0], self.scale[1])
        img = np.array(img)
        transform = A.RandomToneCurve(scale=scale, always_apply=self.always_apply, p=self.p)
        new_img = transform(image=img)['image']
        new_img = Image.fromarray(new_img)
        return new_img

class RandomErase(object):
    def __init__(self, choices=[0,1,'random']):
        self.choices = choices
    def __call__(self, image):
        image = image.convert('RGB')
        choice = random.choice(self.choices)
        t = transforms.Compose([
            transforms.ToTensor(),
        ])
        x = t(image).unsqueeze(0)
        x_1=transforms.RandomErasing(p=1, value=choice)(x)
        return transforms.ToPILImage()(x_1[0])
    
class AutoAug(object):
    def __call__(self, image):
        image = image.convert('RGB')
        return transforms.AutoAugment()(image)
    
class AutoSeg(object):
    def show_anns(self,anns):
        if len(anns) == 0:
            return
        sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
        ax = plt.gca()
        ax.set_autoscale_on(False)

        img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
        img[:,:,3] = 0
        for ann in sorted_anns:
            m = ann['segmentation']
            color_mask = np.concatenate([np.random.random(3), [0.35]])
            img[m] = color_mask
        ax.imshow(img)
    
    def __call__(self, image):
        image = image.convert('RGB')
        image = cv2.cvtColor(np.array(image.resize((256,256))), cv2.COLOR_RGB2BGR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
        sam_checkpoint = "sam_vit_b_01ec64.pth"
        model_type = "vit_b"
        device = "cpu"

        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=device)
        mask_generator = SamAutomaticMaskGenerator(sam)
        masks = mask_generator.generate(image)
        
        plt.figure(figsize=(10,10))
        plt.imshow(image)
        self.show_anns(masks)
        plt.axis('off')
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        plt.show()
        im = Image.open(buf)
        return im.convert('RGB').resize((256,256))
    
class FGRemove(object):
    def __call__(self, image):
        image = image.convert('RGB')
        image_r = remove(image).convert('RGB')
        return Image.fromarray(np.array(image) - np.array(image_r))
    
class ErodeORDilate(object):
    def __init__(self, choice=1, kernel = [3, 20]):
        self.choice = choice
        self.kernel = kernel
    
    def apply_erosion(self, image, kernel_size=3, iterations=1):
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        eroded_image = cv2.erode(image, kernel, iterations=iterations)
        return eroded_image

    def apply_dilation(self, image, kernel_size=3, iterations=1):
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        dilated_image = cv2.dilate(image, kernel, iterations=iterations)
        return dilated_image
    
    def __call__(self, image):
        image = image.convert('RGB')
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        choice = random.randint(0, self.choice)
        kernel = random.randint(self.kernel[0], self.kernel[1])
        if choice == 0:
            eroded_image = self.apply_erosion(image, kernel_size=kernel, iterations=1)
            return Image.fromarray(cv2.cvtColor(eroded_image, cv2.COLOR_BGR2RGB))
        else:
            dilated_image = self.apply_dilation(image, kernel_size=kernel, iterations=1)
            return Image.fromarray(cv2.cvtColor(dilated_image, cv2.COLOR_BGR2RGB))
        
class Fleet(object):
    '''
    The term Fleet is used to describe the passage of time or years that flow like water. 
    In the context of image processing, it specifically refers to the transformation of the original image 
    into one with a sense of time or the accumulation of years, as shown in the effect demonstrated in the image.
    '''
    def __init__(self, number=[10,20]):
        self.number = number
        
    def __call__(self, image):
        image = image.convert('RGB')
        number = random.randint(self.number[0], self.number[1])
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        rows, cols = image.shape[:2]
        dst = np.zeros((rows, cols, 3), dtype="uint8")
        for i in range(rows):
            for j in range(cols):
                B = math.sqrt(image[i,j][0]) * number
                G =  image[i,j][1]
                R =  image[i,j][2]
                if B>255:
                    B = 255
                dst[i,j] = np.uint8((B, G, R)) 
        return Image.fromarray(cv2.cvtColor(dst, cv2.COLOR_BGR2RGB))
    

class WaterWave(object):
    def __init__(self, wavelength = [20,60], amplitude = [30, 100], centreX = [0, 1], centreY = [0, 1]):
        self.wavelength = wavelength
        self.amplitude = amplitude
        self.centreX = centreX
        self.centreY = centreY
        
    def __call__(self, image):
        image = image.convert('RGB')
        wavelength = random.randint(self.wavelength[0], self.wavelength[1])
        amplitude = random.randint(self.amplitude[0], self.amplitude[1])
        centreX = random.uniform(self.centreX[0], self.centreX[1])
        centreY = random.uniform(self.centreY[0], self.centreY[1])
        
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        rows, cols = image.shape[:2]
        dst = np.zeros((rows, cols, 3), dtype="uint8")
        phase = math.pi / 4
        radius = min(rows, cols) / 2
        icentreX = cols*centreX
        icentreY = rows*centreY
        for i in range(rows):
            for j in range(cols):
                dx = j - icentreX
                dy = i - icentreY
                distance = dx*dx + dy*dy

                if distance>radius*radius:
                    x = j
                    y = i
                else:
                    #计算水波区域
                    distance = math.sqrt(distance)
                    amount = amplitude * math.sin(distance / wavelength * 2*math.pi - phase)
                    amount = amount *  (radius-distance) / radius
                    amount = amount * wavelength / (distance+0.0001)
                    x = j + dx * amount
                    y = i + dy * amount

                #边界判断
                if x<0:
                    x = 0
                if x>=cols-1:
                    x = cols - 2
                if y<0:
                    y = 0
                if y>=rows-1:
                    y = rows - 2

                p = x - int(x)
                q = y - int(y)

                #图像水波赋值
                dst[i, j, :] = (1-p)*(1-q)*image[int(y),int(x),:] + p*(1-q)*image[int(y),int(x),:]
                + (1-p)*q*image[int(y),int(x),:] + p*q*image[int(y),int(x),:]
                
        return Image.fromarray(cv2.cvtColor(dst, cv2.COLOR_BGR2RGB))
    
class Binary(object):
    def __init__(self, choice=4):
        self.choice = choice
        
    def __call__(self, image):
        image = image.convert('RGB')
        choice = random.randint(0, self.choice)
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        (T, thresh) = cv2.threshold(blurred, 155, 255, cv2.THRESH_BINARY)
        (T, threshInv) = cv2.threshold(blurred, 155, 255, cv2.THRESH_BINARY_INV)

        if choice == 0:
            ret, thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY) 
        elif choice == 1:
            ret, thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
        elif choice == 2:
            ret, thresh = cv2.threshold(image, 127, 255, cv2.THRESH_TRUNC)
        elif choice == 3:
            ret, thresh = cv2.threshold(image, 127, 255, cv2.THRESH_TOZERO)
        else:
            ret, thresh = cv2.threshold(image, 127, 255, cv2.THRESH_TOZERO_INV)

        return Image.fromarray(cv2.cvtColor(thresh, cv2.COLOR_BGR2RGB))
        
class Dehaze(object):
    def DarkChannel(self, im,sz):
        b,g,r = cv2.split(im)
        dc = cv2.min(cv2.min(r,g),b);
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(sz,sz))
        dark = cv2.erode(dc,kernel)
        return dark

    def AtmLight(self, im,dark):
        [h,w] = im.shape[:2]
        imsz = h*w
        numpx = int(max(math.floor(imsz/1000),1))
        darkvec = dark.reshape(imsz);
        imvec = im.reshape(imsz,3);

        indices = darkvec.argsort();
        indices = indices[imsz-numpx::]

        atmsum = np.zeros([1,3])
        for ind in range(1,numpx):
            atmsum = atmsum + imvec[indices[ind]]

        A = atmsum / numpx;
        return A

    def TransmissionEstimate(self, im,A,sz):
        omega = 0.95;
        im3 = np.empty(im.shape,im.dtype);

        for ind in range(0,3):
            im3[:,:,ind] = im[:,:,ind]/A[0,ind]

        transmission = 1 - omega*self.DarkChannel(im3,sz);
        return transmission

    def Guidedfilter(self, im,p,r,eps):
        mean_I = cv2.boxFilter(im,cv2.CV_64F,(r,r));
        mean_p = cv2.boxFilter(p, cv2.CV_64F,(r,r));
        mean_Ip = cv2.boxFilter(im*p,cv2.CV_64F,(r,r));
        cov_Ip = mean_Ip - mean_I*mean_p;

        mean_II = cv2.boxFilter(im*im,cv2.CV_64F,(r,r));
        var_I   = mean_II - mean_I*mean_I;

        a = cov_Ip/(var_I + eps);
        b = mean_p - a*mean_I;

        mean_a = cv2.boxFilter(a,cv2.CV_64F,(r,r));
        mean_b = cv2.boxFilter(b,cv2.CV_64F,(r,r));

        q = mean_a*im + mean_b;
        return q;

    def TransmissionRefine(self, im,et):
        gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY);
        gray = np.float64(gray)/255;
        r = 60;
        eps = 0.0001;
        t = self.Guidedfilter(gray,et,r,eps);

        return t;

    def Recover(self, im,t,A,tx = 0.1):
        res = np.empty(im.shape,im.dtype);
        t = cv2.max(t,tx);

        for ind in range(0,3):
            res[:,:,ind] = (im[:,:,ind]-A[0,ind])/t + A[0,ind]

        return res
    
    def __call__(self, image):

        image = image.convert('RGB')
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        I = image.astype('float64')/255;
        dark = self.DarkChannel(I,15);
        A = self.AtmLight(I,dark);
        te = self.TransmissionEstimate(I,A,15);
        t = self.TransmissionRefine(image,te);
        J = self.Recover(I,t,A,0.1);
        
        return Image.fromarray(cv2.cvtColor((J*255).astype(np.uint8), cv2.COLOR_BGR2RGB))
    
class ImagePyramid(object):
    def __init__(self, levels = [2, 10], weight = [1.2, 4]):
        self.levels = levels
        self.weight = weight
    
    def gaussian_pyramid(self, image, levels, weight):
        pyramid = [image]  
        for i in range(levels - 1):
            if int(image.width / weight) >0 and int(image.height / weight) >0:
                image = image.resize((int(image.width / weight), int(image.height / weight)))
            else:
                image = image
            pyramid.append(image)  
        return pyramid
    
    def merge_pyramid_images_centered(self, pyramid_images):
        base_width, base_height = pyramid_images[0].size
        
        merged_image = Image.new("RGB", (base_width, base_height), color=(0, 0, 0))

        for img in pyramid_images:
            img_width, img_height = img.size

            position_x = (base_width - img_width) // 2
            position_y = (base_height - img_height) // 2

            merged_image.paste(img, (position_x, position_y))

        return merged_image
    
    def __call__(self, image):
        image = image.convert('RGB')
        levels = random.randint(self.levels[0],self.levels[1])
        weight = random.uniform(self.weight[0],self.weight[1])
        gaussian_pyramid_im = self.gaussian_pyramid(image, levels, weight)
        merged_image = self.merge_pyramid_images_centered(gaussian_pyramid_im)

        return merged_image

class Swirl(object):
    def __init__(self, degree = [10, 50], MidX = [0.2,0.8], MidY = [0.2,0.8]):
        self.degree = degree
        self.MidX = MidX
        self.MidY = MidY
    
    def __call__(self, image):
        image = image.convert('RGB')
        I = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        (W, H, chanel) = I.shape
        dst_data = np.zeros((W, H, chanel), dtype=np.uint8)
        
        degree = random.randint(self.degree[0], self.degree[1])
        swirldegree = degree / 1000.0
        MidX = W*random.uniform(self.MidX[0], self.MidX[1])
        MidY = H*random.uniform(self.MidY[0], self.MidY[1])
        
        for y in range(H):
            for x in range(W):
                Yoffset = y - MidY
                Xoffset = x - MidX
                radian = atan2(Yoffset, Xoffset)
                radius = sqrt(Xoffset * Xoffset + Yoffset * Yoffset)

                X = int(radius * cos(radian + radius * swirldegree) + MidX)
                Y = int(radius * sin(radian + radius * swirldegree) + MidY)

                if X >= W:
                    X = W - 1
                if X < 0:
                    X = 0
                if Y >= H:
                    Y = H - 1
                if Y < 0:
                    Y = 0

                dst_data[x, y, :] = I[X, Y, :]
                
        return Image.fromarray(cv2.cvtColor((dst_data).astype(np.uint8), cv2.COLOR_BGR2RGB))
    
class PatchInterleave(object):
    
    def interleave_patches(self, imageA, imageB, m, n):
        width, height = imageA.size
        patch_width = width // m
        patch_height = height // n

        result = Image.new("RGB", (width, height))

        for i in range(m):
            for j in range(n):
                box = (i * patch_width, j * patch_height, (i + 1) * patch_width, (j + 1) * patch_height)
                if (j % 2 == 0 and i % 2 == 0) or (j % 2 == 1 and i % 2 == 1):
                    source = imageA
                else:
                    source = imageB
                patch = source.crop(box)
                result.paste(patch, (i * patch_width, j * patch_height))

        return result
    
    def __init__(self, m = [4, 20], n = [4, 20], path = 'gsdata/VSC/data/training_images_9/'):
        self.m = m
        self.n = n
        self.path = path
        self.ls = os.listdir(path)
    
    def __call__(self, image):
        image = image.convert('RGB')
        m = random.randint(self.m[0], self.m[1])
        n = random.randint(self.n[0], self.n[1])
        imageA = image
        imageB_path = self.path + '/' + random.choice(self.ls)
        imageB = Image.open(imageB_path).resize((imageA.size[0], imageA.size[1]))
        result = self.interleave_patches(imageA, imageB, m, n)
        
        return result
    
