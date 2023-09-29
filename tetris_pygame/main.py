from settings import *
from tetris import Tetris, Text, CamView
import pathlib
import pygame, sys
from button import Button
import pygame.camera


class App:
    def __init__(self):
        pg.init()
        pg.display.set_caption('Tetris')
        self.screen = pg.display.set_mode(WIN_RES)
        self.clock = pg.time.Clock()
        self.set_timer()
        self.images = self.load_images()
        self.tetris = Tetris(self)
        self.text = Text(self)
        self.cam = CamView(self)

    def load_images(self):
        files = [item for item in pathlib.Path(SPRITE_DIR_PATH).rglob('*.png') if item.is_file()]
        images = [pg.image.load(file).convert_alpha() for file in files]
        images = [pg.transform.scale(image, (TILE_SIZE, TILE_SIZE)) for image in images]
        return images

    def set_timer(self):
        self.user_event = pg.USEREVENT + 0
        self.fast_user_event = pg.USEREVENT + 1
        self.anim_trigger = False
        self.fast_anim_trigger = False
        pg.time.set_timer(self.user_event, ANIM_TIME_INTERVAL)
        pg.time.set_timer(self.fast_user_event, FAST_ANIM_TIME_INTERVAL)

    def update(self):
        self.tetris.update()
        self.clock.tick(FPS)



    #DRAW FEATURES ON SCREEN
    def draw(self):
        self.screen.fill(color=BG_COLOR)
        self.screen.fill(color=FIELD_COLOR, rect=(0, 0, *FIELD_RES))
        self.tetris.draw()
        self.text.draw()
        #Place Camera onto game Screen
        self.cam.draw_camera()
        pg.display.flip()
    

    def check_events(self):
        self.anim_trigger = False
        self.fast_anim_trigger = False
        for event in pg.event.get():
            if event.type == pg.QUIT or (event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE):
                pg.quit()
                sys.exit()
            elif event.type == pg.KEYDOWN:
                self.tetris.control(pressed_key=event.key)
            elif event.type == self.user_event:
                self.anim_trigger = True
            elif event.type == self.fast_user_event:
                self.fast_anim_trigger = True

    def run(self):
        while True:
            self.check_events()
            self.update()
            self.draw()


if __name__ == '__main__':
    # app = App()
    # app.run()
    
    pygame.init()
    pygame.display.set_caption("Menu")

    SCREEN = pygame.display.set_mode((WIN_W, WIN_H), pygame.RESIZABLE)
    
    #Initalise Camera
    pygame.camera.init()

    #See if Computer Has a Camera
    clist = pygame.camera.list_cameras()
    if not clist:
        raise ValueError("Sorry, no cameras detected.")

    camsize = (640,480)
    webcam = pygame.camera.Camera(clist[0], camsize)
    webcam.start()
    webcam.set_controls(hflip = True, vflip = False)

    BG = pygame.image.load("assets/bg_img/spacebgg.jpeg") 

    def get_font(size): # Returns Press-Start-2P in the desired size
        return pygame.font.Font("assets/font/font.ttf", size)

    def main_camera(SCREEN,center_width,main_height):
        #Camera Setup
        img = webcam.get_image();
        SCREEN.blit(img,(center_width,main_height))



    def main_menu():
        center_width = WIN_W*0.5

        while True:
            SCREEN.blit(BG, (0, 0))

            MENU_MOUSE_POS = pygame.mouse.get_pos()

            MENU_TEXT = get_font(50).render("TETRIS MENU", True, "#b68f40")
            MENU_RECT = MENU_TEXT.get_rect(center=((center_width), (WIN_H*0.05)))

            PLAY_BUTTON = Button(image=pygame.image.load("assets/menu/Play Rect.png"), pos=(center_width, (WIN_H*0.76)), 
                                text_input="PLAY", font=get_font(75), base_color="#d7fcd4", hovering_color="White")
        
            QUIT_BUTTON = Button(image=pygame.image.load("assets/menu/Quit Rect.png"), pos=(center_width, (WIN_H*0.92)), 
                                text_input="QUIT", font=get_font(75), base_color="#d7fcd4", hovering_color="White")
            
            main_camera(SCREEN,WIN_W*0.28,WIN_H*0.10)
            

            SCREEN.blit(MENU_TEXT, MENU_RECT)

            for button in [PLAY_BUTTON, QUIT_BUTTON]:
                button.changeColor(MENU_MOUSE_POS)
                button.update(SCREEN)
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if PLAY_BUTTON.checkForInput(MENU_MOUSE_POS):
                        pygame.quit()
                        app = App()
                        app.run()
                    if QUIT_BUTTON.checkForInput(MENU_MOUSE_POS):
                        pygame.quit()
                        sys.exit()

            pygame.display.update()

    main_menu()