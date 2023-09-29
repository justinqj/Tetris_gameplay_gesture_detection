#Web Cam Pygame Testing
#https://www.youtube.com/watch?v=1cORHfg7g7g
#https://www.pygame.org/docs/tut/CameraIntro.html

import pygame
import pygame.camera
import cv2
import pygame.image
import time
import datetime


pygame.init()
pygame.camera.init()

webcam = pygame.camera.Camera(('FaceTime HD Camera'))

#webcam.set_resolution(20,20)
screen = pygame.display.set_mode((1000,1000))
pygame.display.set_caption("PYCAM")
webcam.start()
webcam.set_controls(hflip = True, vflip = False)

#Mac WebCam
#print(pygame.camera.list_cameras())
#['FaceTime HD Camera']


done= False
while not done:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done=True
    

    keypressed = pygame.key.get_pressed()
    img = webcam.get_image();
    if keypressed[pygame.K_SPACE]:
        print("TOOK IMAGE")
        current_time = now = datetime.datetime.now()
        pygame.image.save(img,f"webcam/mypic_{current_time}.jpeg")
    
    img = webcam.get_image();
    screen.blit(img,(0,0))

    pygame.display.update()


    