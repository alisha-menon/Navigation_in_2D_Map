import pygame
import random
import numpy as np
import os

from modules.hd_module import hd_module

class game_module:
    def __init__(self):
        self.world_size = (10,10)
        self.grid_size = (self.world_size[0]+2, self.world_size[1]+2)
        self.scale = 50
        self.pixel_dim = (self.grid_size[0]*self.scale, self.grid_size[1]*self.scale)

        self.num_obs = 15
        self.timeout = 100

        self.white = (255,255,255)
        self.blue = (0,0,225)
        self.green = (0,255,0)
        self.black = (0,0,0)

        self.pos = [0,0]
        self.goal_pos = [0,0]
        self.obs = []
        self.obs_mat = np.zeros(self.world_size)

        self.steps = 0

        self.hd_module = hd_module()
        self.num_cond = self.hd_module.num_cond
        self.num_thrown = self.hd_module.num_thrown

        self.outdir = './data/'
        self.outfile = self.outdir + 'game_datanirudhrandom.out'
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)


    def setup_game(self):
        self.obs = []
        self.obs_mat = np.zeros(self.world_size)
        num_block = self.world_size[0]*self.world_size[1]
        obs_idx = random.sample(list(range(num_block)), self.num_obs+1)
        for i in range(self.num_obs):
            row_pos = obs_idx[i]//self.world_size[0]
            col_pos = obs_idx[i]%self.world_size[1]
            self.obs.append((row_pos, col_pos))
            self.obs_mat[row_pos,col_pos] = 1

        self.pos = [obs_idx[-1]//self.world_size[0], obs_idx[-1]%self.world_size[1]]
        self.random_goal_location()
        self.steps = 0
        return

    def train_from_file(self, filename):
        self.hd_module.train_from_file(filename)
        self.num_cond = self.hd_module.num_cond
        self.num_thrown = self.hd_module.num_thrown

    def play_game(self, gametype):
        pygame.init()
        screen = pygame.display.set_mode(self.pixel_dim)


        f = open(self.outfile, 'w')

        running = True
        not_crash = True

        actuator = 0
        while running:
            self.setup_game()
            while not_crash:
                self.game_step(gametype, screen)
                pygame.display.update()

                event = pygame.event.wait()
                if event.type == pygame.QUIT:
                    not_crash = False
                    running = False
                elif event.type == pygame.KEYDOWN:
                    current_sensor = self.get_sensor()
                    current_sensor.append(actuator)
                    if event.key == pygame.K_LEFT:
                        self.pos[0] -= 1
                        actuator = 0
                    elif event.key == pygame.K_RIGHT:
                        self.pos[0] += 1
                        actuator = 1
                    elif event.key == pygame.K_UP:
                        self.pos[1] -= 1
                        actuator = 2
                    elif event.key == pygame.K_DOWN:
                        self.pos[1] += 1
                        actuator = 3
                    elif event.key == pygame.K_RETURN:
                        self.setup_game()


                    if (self.check_collision(self.pos[0], self.pos[1])):
                        not_crash = False
                    else:
# *********************** CHANGE BASED ON SENSOR DATA *************************
                        sensor_str = "{}, {}, {}, {}, {}, {}, {}".format(*current_sensor)
                        f.write(sensor_str + ", " + str(actuator) + "\n")
# *****************************************************************************
                self.game_step(gametype, screen)
                pygame.display.update()

            event2 = pygame.event.wait()
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                not_crash = True


        pygame.display.quit()
        pygame.quit()
        f.close()
        return

    def set_sensor_weight(self,sensor_weight):
        self.hd_module.sensor_weight = sensor_weight
        return

    def set_threshold_known(self,threshold_known):
        self.hd_module.threshold_known = threshold_known
        return

    def set_softmax_param(self,softmax_param):
        self.hd_module.softmax_param = softmax_param
        return

    def autoplay_game(self, gametype):
        pygame.init()
        screen = pygame.display.set_mode(self.pixel_dim)
        clock = pygame.time.Clock()
        running = True
        not_crash = True

        last_act = 0
        while running:
            self.setup_game()
            self.steps = 0
            while not_crash:
                self.game_step(gametype, screen)
                pygame.display.update()

                clock.tick(3)
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                        not_crash = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_RETURN:
                            self.setup_game()

                current_sensor = self.get_sensor()
                current_sensor.append(last_act)
                act_out = self.hd_module.test_sample(current_sensor)
                if act_out == 0:
                    self.pos[0] -= 1
                elif act_out == 1:
                    self.pos[0] += 1
                elif act_out == 2:
                    self.pos[1] -= 1
                elif act_out == 3:
                    self.pos[1] += 1

                last_act = act_out
                if (self.check_collision(self.pos[0], self.pos[1])):
                    not_crash = False
                    print(not_crash)
                if (self.steps >= self.timeout):
                    not_crash = False
                    print(not_crash)

                self.steps += 1

                self.game_step(gametype, screen)
                pygame.display.update()

            event2 = pygame.event.wait()
            if event2.type == pygame.QUIT:
                running = False
            elif event2.type == pygame.KEYDOWN:
                not_crash = True

        pygame.display.quit()
        pygame.quit()
        return

    def test_game(self, num_test):
        not_crash = True

        last_act = 0

        success = 0
        crash = 0
        stuck = 0
        for i in range(num_test):
            not_crash = True
            self.setup_game()
            self.steps = 0
            while not_crash:

                if self.goal_pos == self.pos:
                    self.random_goal_location()
                    success += 1
                    break

                current_sensor = self.get_sensor()
                current_sensor.append(last_act)
                act_out = self.hd_module.test_sample(current_sensor)
                if act_out == 0:
                    self.pos[0] -= 1
                elif act_out == 1:
                    self.pos[0] += 1
                elif act_out == 2:
                    self.pos[1] -= 1
                elif act_out == 3:
                    self.pos[1] += 1

                last_act = act_out
                if (self.check_collision(self.pos[0], self.pos[1])):
                    not_crash = False
                    crash += 1
                elif (self.steps >= self.timeout):
                    not_crash = False
                    stuck += 1

                self.steps += 1


        print("success: {} \t crash: {} \t stuck: {}".format(success, crash, stuck))
        print("success rate: {:.2f}".format(success/(success+crash+stuck)))

        return success,crash,stuck

    def game_step(self, gametype, screen):
        screen.fill(self.white)
        self.draw_walls(screen)
        self.draw_obstacles(screen)
        self.draw_me(screen)
        if (gametype):
            if self.goal_pos == self.pos:
                self.random_goal_location()
                self.steps = 0
            self.draw_goal(screen)
        return

    def draw_me(self, screen):
        xpixel = (self.pos[0]+1)*self.scale
        ypixel = (self.pos[1]+1)*self.scale
        pygame.draw.rect(screen, self.blue, [xpixel,ypixel,self.scale,self.scale])
        return


    def draw_obstacles(self, screen):
        for pos in self.obs:
            xpos = (pos[0]+1)*self.scale
            ypos = (pos[1]+1)*self.scale
            pygame.draw.rect(screen, self.black, [xpos,ypos,self.scale,self.scale])
        return

    def draw_goal(self, screen):
        xpixel = (self.goal_pos[0]+1)*self.scale
        ypixel = (self.goal_pos[1]+1)*self.scale
        pygame.draw.rect(screen, self.green, [xpixel,ypixel,self.scale,self.scale])
        return

    def draw_walls(self, screen):
        pygame.draw.rect(screen, self.black, [0,0,self.pixel_dim[0]-self.scale,self.scale])
        pygame.draw.rect(screen, self.black, [self.pixel_dim[0]-self.scale,0,self.scale,self.pixel_dim[1]-self.scale])
        pygame.draw.rect(screen, self.black, [self.scale,self.pixel_dim[1]-self.scale,self.pixel_dim[0]-self.scale,self.scale])
        pygame.draw.rect(screen, self.black, [0,self.scale,self.scale,self.pixel_dim[1]-self.scale])
        return

    def pos_oob(self, xpos, ypos):
        # Check if (xpos,ypos) is out of bounds
        oob = 0
        if (xpos < 0 or xpos >= self.world_size[0]):
            oob = 1
        if (ypos < 0 or ypos >= self.world_size[1]):
            oob = 1
        return oob

    def check_collision(self, xpos, ypos):
        # Check if (xpos,ypos) is out of bounds or occupied by object
        collision = 0
        if (self.pos_oob(xpos, ypos)):
            collision = 1
            #print(collision)
        else:
            if (self.obs_mat[xpos, ypos]):
                collision = 1
                #print(collision)
        return collision

    def random_goal_location(self):
        # Choose random unoccupied square for the goal position
        num_block = self.world_size[0]*self.world_size[1]
        goal_idx = random.randrange(num_block)
        row_pos = goal_idx//self.world_size[0]
        col_pos = goal_idx%self.world_size[1]
        while (self.check_collision(row_pos,col_pos)):
            goal_idx = random.randrange(num_block)
            row_pos = goal_idx//self.world_size[0]
            col_pos = goal_idx%self.world_size[1]
        self.goal_pos = [row_pos, col_pos]
        return


# *********************** CHANGE BASED ON SENSOR DATA *************************
    def get_sensor(self):
        # list of coordinates for squares around current position
        sensor_pos = [(self.pos[0]-1, self.pos[1]),
                (self.pos[0]+1, self.pos[1]),
                (self.pos[0], self.pos[1]-1),
                (self.pos[0], self.pos[1]+1)]
        sensor_vals = [self.check_collision(xpos,ypos) for (xpos,ypos) in sensor_pos]
        delta_x = self.goal_pos[0] - self.pos[0]
        delta_y = self.goal_pos[1] - self.pos[1]
        sensor_vals.extend([delta_x, delta_y])
        return sensor_vals
# *****************************************************************************
