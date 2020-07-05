import pygame
from PIL import Image, ImageOps
import cv2
import numpy as np

import time

from keras.models import load_model

screen_width = screen_height = 504
final_dim = 28
length = 50
BLACK = pygame.Color(0,0,0)
WHITE = pygame.Color(255,255,255)
FONT = ('verdana', 14)

'''
Images in mnist dataset are black background with white writing so must convert all 0s to 1s and 1s to 0s
Invert image color
'''



def conv(image):
    img = Image.open(image)
    im_invert = ImageOps.invert(img)
    im_invert.save('screenshot.jpg', quality=99)


'''
Canvas size is initially ~20x the mnist data image size so must reduce the img size
'''
def resize(image):
    img = Image.open(image)
    new_image = img.resize((28,28))
    new_image.save('screenshot.jpg')
    conv('screenshot.jpg')
    imgGray = cv2.imread('screenshot.jpg', cv2.IMREAD_GRAYSCALE)
    finImage = np.reshape(imgGray,(1,784))
    return finImage



def main():
    global window

    run = True
    pygame.init()

    window = pygame.display.set_mode((screen_width, screen_height))
    window.fill(WHITE)
    mouse = pygame.mouse
    pygame.display.set_caption("Number Guesser")

    while run:
        pygame.time.delay(15)
        left_pressed, middle_pressed, right_pressed = mouse.get_pressed()


        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

            elif left_pressed:
                (x,y) = mouse.get_pos()
                pygame.draw.rect(window,BLACK,(x,y,length,length))

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    pygame.key.set_repeat(500, 500)
                    pygame.image.save(window, 'screenshot.jpg')
                    image = resize('screenshot.jpg')
                    new_model = load_model('model.h5')
                    prediction = new_model.predict(image)
                    msg = np.argmax(prediction, axis=1)
                    print('I predict this number to be ' + str(msg))
                    window.fill(WHITE)
                if event.key == pygame.K_q:
                    run = False

        pygame.display.update()


if __name__ == "__main__":
    main()