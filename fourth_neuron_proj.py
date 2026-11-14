import pygame
import numpy as np
import math
from math import dist
G = 6.67430e-11
MAX_GAS = 5
MIN_GAS = 0
MAX_ANGEL_STEP = 5
RADIUS_LEN = 50 #math.hypot(750,750)
ANGLE = 30
LINES_NUM = 7
W_LIM = 3
B_LIM = 3
HIDDEN_LAYER_SIZES = [12,5]

ROCKET_IMG = None
MOON_GROUND_IMG = None
FPS = 90
ROCKET_MASS = 5 #KG
PX_PER_METER = 5





# MARK: Neuron
class Neuron:
    def __init__(self, weights:list, bias):
        self.bias = bias
        self.weights = weights
    def predict(self, inputs):
        '''returns the answer in some number format might be float64'''
        total = 0
        for ipnt,weight in zip(inputs,self.weights):
            total += ipnt*weight
        total += self.bias
        result = np.tanh(total)
        return result
    def get_gens(self) -> list[list[float] | float]:
        return [self.weights,self.bias]
    def set_gens(self, gens:list):
        "given a list or tuple in the protocol [ weights:list, bias]"
        weights_stats = gens[0]
        for i in range(len(self.weights)):
            self.weights[i] = weights_stats[i]
        self.bias = gens[1]
# MARK: Layer
class Layer():
    def __init__(self, size, previous_layer_size, neurons=None):
        self.size = size
        if neurons is None:
            self.neurons = [Neuron([uniform(-W_LIM,W_LIM) for i in range(previous_layer_size)],uniform(-B_LIM,B_LIM)) for i in range(size)]
        else:
            self.neurons = neurons
    def predict(self, inputs):
        layer_output = []
        for n in self.neurons:
            layer_output.append(n.predict(inputs))
        return layer_output
    def get_gens(self):
        return [neuron.get_gens() for neuron in self.neurons]
    def set_gens(self,gens:list):# gens is [n1.g,n2.g...]
        for n_gens, neuron in zip(gens, self.neurons):
            neuron.set_gens(n_gens)

# MARK: Brain
# added input! and Layer class and pre_lay_size!
class Brain:
    def __init__(self,input_lay_size, hid_lay_sizes:list[int], output_layer_size):
        '''input size is the amount of inputs the brain gets, than hidden layers sizes -> [10,10] -> (two hidden layers of size 10), than output layer size so Brain(3,5,5,2) is inp3,hid5,hid5,out2'''
        # creating the layers (middle, output):
        # creating the list of the hidden layers -> list[Layer]
        first_hidden_layer = [Layer(hid_lay_sizes[0],input_lay_size)]
        self.hidden_layers:list[Layer] =  first_hidden_layer + [Layer(hid_lay_sizes[i],hid_lay_sizes[i-1]) for i in range(1,len(hid_lay_sizes))]
        self.output_layer = Layer(output_layer_size, hid_lay_sizes[-1])
        self.fitness = 0
    def soft_inputs(self, inputs):
        soften_inputs = []
        for inp in inputs:
            soften_inputs.append(inp/RADIUS_LEN if inp>=0 else 0)
        return soften_inputs
    def predict(self,inputs):
        inputs = self.soft_inputs(inputs)
        result_of_pre_lay = inputs
        for hid_lay in self.hidden_layers:
            result_of_pre_lay = hid_lay.predict(result_of_pre_lay)
        inputs_for_output_lay = result_of_pre_lay
        final_output = self.output_layer.predict(inputs_for_output_lay)
        return final_output
    def get_gens(self):
        gens = []
        # gen = []
        for hid_lay in self.hidden_layers:
            gens.append(hid_lay.get_gens())
        # gen = [list_of_hidden_layers_gens] -> [[n1.g,n2.g,n3.g],[n1.g,n2.g,n3.g]]
        gens.append(self.output_layer.get_gens())
        return gens
    def set_gens(self, gens):
        '''give protocol -> [[n1.g,n2.g,n3.g],[n1.g,n2.g,n3.g]]'''
        hidden_layers_gens = gens[:-1] # list of list of every layer gens [[n1.g,n2.g,n3.g],[n1.g,n2.g,n3.g]]
        output_layer_gens= gens[-1] # list of of neurons gens [n1.g,n2.g...]

        for hid_lay_gens,hid_lay in zip(hidden_layers_gens,self.hidden_layers):
            hid_lay.set_gens(hid_lay_gens)
        self.output_layer.set_gens(output_layer_gens)
    def __str__(self):
        return str(self.get_gens())

# MARK: Evolution
class Evolution:
    mutation_chance = 1/5
    @staticmethod
    def child_neuron(n1:Neuron, n2:Neuron) -> list:
        weights = []
        for w1, w2 in zip(n1[0],n2[0]):
            choosen_w = choice([w1,w2])
            choosen_w = (choosen_w + uniform(-0.4,0.4)) if np.random.rand() < Evolution.mutation_chance else choosen_w
            weights.append(choosen_w)
        choosen_b = choice([n1[1], n2[1]])
        choosen_b = (choosen_b + uniform(-0.4,0.4)) if np.random.rand() < Evolution.mutation_chance else choosen_b
        bias = choosen_b
        return [weights,bias]
    @staticmethod
    def child_brain(brain1:Brain, brain2:Brain) -> list[list[Neuron]]:
        '''returns [ mid layer of the child that came out, ditto to the output layer]'''
        brain1 = brain1.get_gens()
        brain2 = brain2.get_gens()
        # creating zip for each layer of the two brains
        hid_lays = zip(brain1[:-1], brain2[:-1]) # brain[:-1] a list of Layer.get_gens() [l1.g, l2.g]
        out_lay = zip(brain1[-1], brain2[-1]) # brain[-1] is a list layer.get_gens() l1.g
        new_hid_lays = []
        new_hid_lay = []
        for LG_mom, LG_dad in hid_lays:
            new_hid_lay = []
            for n_mom, n_dad in zip(LG_mom,LG_dad):
                n_child = Evolution.child_neuron(n_mom,n_dad)
                new_hid_lay.append(n_child)
            new_hid_lays.append(new_hid_lay)
        new_out_lay = []
        for n_mom, n_dad in out_lay:
            n_child = Evolution.child_neuron(n_mom,n_dad)
            new_out_lay.append(n_child)    
        result = new_hid_lays
        result.append(new_out_lay)       
        return result
    def next_gen(brains:list[Brain]) -> list[Brain]:
        '''returns a list of object Brain of next gen'''
        #print(f"DEBUG: first brain gens (best one) : {brains[0].get_gens()}")

        next_gen_gens = []
        for i in range(len(brains[:len(brains)//2])-2):
            brain_daddy = brains[i]
            brain_mommy = brains[i+1]
            child_brain1 = Evolution.child_brain(brain_daddy, brain_mommy)
            child_brain2 = Evolution.child_brain(brain_daddy, brain_mommy)
            next_gen_gens.extend([child_brain1, child_brain2])
        #print(f"DEBUG: amount of new brains (childs only) : {len(next_gen_gens)}")
        next_gen_gens.extend([b.get_gens() for b in brains[:4]])
        #print(f"DEBUG: amount of new brains : {len(next_gen_gens)}")
        next_gen_brains = []
        for brain_gens in next_gen_gens:
            brain1 = Brain(LINES_NUM,HIDDEN_LAYER_SIZES,2)
            brain1.set_gens(brain_gens)
            next_gen_brains.append(brain1)
        return next_gen_brains

# MARK: ||||||||||||||||||||||||||||||||

        
class Vector:
    def __init__(self, angle=0,scalar=0):
        self.angle = angle
        self.scalar = scalar
        self.sin_vec = 0
        self.cos_vec = 0
    def set_angle(self,rotation_size):
        self.angle += rotation_size
        while self.angle<0:
            self.angle = 360 + self.angle
        self.angle %= 360
        return self.angle

    # static because maybe earth gravity and this shit you know maybe not...
    def seperate_verctor(self):
        z,q = 1,-1
        self.angle %= 360
        '''if 180<self.angle<360:
            #because sin(a) for every 360>a>180 is negetive
            q = -1
        if 90<self.angle<270:
            # because cos(a) for every -90<a<90 is negetive
            z = -1'''
        cos_speed = np.cos( Calc.degrees_to_radians(self.angle) )*self.scalar *z
        sin_speed = np.sin( Calc.degrees_to_radians(self.angle) )*self.scalar *q

        self.cos_vec = cos_speed
        self.sin_vec = sin_speed
        return [sin_speed, cos_speed]
    def __str__(self):
        return f"ANGLE : {self.angle} self : {self.scalar}"

class Astro_body:
    def __init__(self, center, radius, mass, color = (200,55,150)):
        self.mass = mass
        self.radius = radius
        self.color = color
        self.center = center
    def draw(self, screen):
        pygame.draw.circle(screen, self.color, self.center, self.radius)
    
    

# MARK: Body
class Body:
    def __init__(self, start_pos = (200,680)):
        self.x = start_pos[0]
        self.y = start_pos[1]
        self.center = (0,0)
        self.MF = 15*ROCKET_MASS
        self.LF = 5
        self.RF = 5
        self.angleM = 0
        self.angleL = 0
        self.angleR = 0
        self.angle = 0
        self.xspeed = 0
        self.yspeed = 0
        self.mainF = False
        self.leftF = False
        self.rightF = False
        self.force_xgrid_vec = 0
        self.force_ygrid_vec = 0
        self.angle_step = 1
        # remember that -> to the right is the 0 degree angle!
        self.vectors = [Vector(90,0),Vector(0,0),Vector(180,0),Vector(270,10*ROCKET_MASS)]
    def rotate(self,rotation_size):
        self.angle += rotation_size*5
    def fire(self, engine_num, answer:bool):
        if answer:
            if engine_num == 1:
                self.vectors[engine_num-1].scalar = self.MF
            elif engine_num == 2:
                self.vectors[engine_num-1].scalar = self.LF
            else:
                self.vectors[engine_num-1].scalar = self.RF
    def reset_vectors(self):
        for i in range(0,3):
            self.vectors[i].scalar = 0 
    def update_angle(self):
        if self.vectors[2].scalar!=0:
            self.angle += self.angle_step
            for i in range(3):
                self.vectors[i].set_angle(self.angle_step)
        if self.vectors[1].scalar != 0:
            self.angle += -self.angle_step
            for i in range(3):
                self.vectors[i].set_angle(-self.angle_step)
    def update_pos(self):
        new_f_xg_v, new_f_yg_v = Calc.resultant_vector(self.vectors)
        self.force_xgrid_vec = new_f_xg_v*PX_PER_METER
        self.force_ygrid_vec = new_f_yg_v*PX_PER_METER
        on_ground = self.ground(693)
        self.reset_vectors()
        ax = self.force_xgrid_vec/ROCKET_MASS
        ay = self.force_ygrid_vec/ROCKET_MASS
        # rule of neuton if sum of forces is zero then static velocity:
        if on_ground and self.yspeed>0: # if self.yspeed > 0 then velocity in y grid is down and can't go down when on ground!
            self.yspeed = 0
        # x = x + vt + 0.5a * (t^2)
        new_x =  ( self.x + self.xspeed*(1/FPS) + 0.5*(ax)*((1/FPS)**2) )
        new_y =  ( self.y + self.yspeed*(1/FPS) + 0.5*(ay)*((1/FPS)**2) )

        self.xspeed = self.xspeed + ax * (1/FPS)
        self.yspeed = self.yspeed + ay * (1/FPS)

        self.x = new_x
        self.y = new_y
        print(new_x,new_y,"|||",self.force_ygrid_vec,self.force_xgrid_vec)
        
    def calc_next(self):
        self.update_angle()
        self.update_pos()
    def ground(self,ground_level):
        if ground_level <= int(self.y):
            if self.force_ygrid_vec>0:
                normal_f = self.force_ygrid_vec
                self.force_ygrid_vec -= normal_f
            return True
        return False
class Calc:
    @ staticmethod
    def resultant_vector(forces):
        total_sin = 0
        total_cos = 0
        for force in forces:
            force.seperate_verctor()
            total_sin += force.sin_vec
            total_cos += force.cos_vec
        return total_cos, total_sin
    @ staticmethod
    def degrees_to_radians(angle):
        return (np.pi/180)*(angle) 
    @ staticmethod
    def law_of_uni_grav(mass1, mass2, dis):
        return G* (mass1*mass2)/(dis**2)
    @ staticmethod
    def force_of_astro_body(rocket:Body, astro_body:Astro_body):
        return Calc.law_of_uni_grav(ROCKET_MASS, astro_body.mass, dist(rocket.center, astro_body.center))
    
# MARK: Window
class Window:
    @staticmethod
    def pygame_init():
        pygame.init()
    @staticmethod
    def load_assests():
        global ROCKET_IMG, MOON_GROUND_IMG
        ROCKET_IMG = pygame.image.load(r"E:\python_1\AI\Fourth_neuron_proj\rocket_ship.png")
        MOON_GROUND_IMG = pygame.image.load(r"E:\python_1\AI\Fourth_neuron_proj\moon_ground.png")

    @staticmethod
    def run(win_size = (900,900)):

        body1 = Body()
        pluto = Astro_body((20,20), 50, 20)
        m,l,r = False,False,False
        
        screen = pygame.display.set_mode(win_size)
        clock = pygame.time.Clock()
        running = True
        while running:
            m,l,r = False,False,False
            screen.fill((2,22,80))
            screen.blit(MOON_GROUND_IMG, (0,693))
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            keys = pygame.key.get_pressed()
            if keys[pygame.K_w]:
                m = True
            if keys[pygame.K_a]:
                r = True
            if keys[pygame.K_d]:
                l = True
            body1.fire(1,m)
            body1.fire(2,l)
            body1.fire(3,r)
            Window.draw_vectors(screen,body1)
            body1.calc_next()
            Window.show_body(screen,body1)
            Window.draw_astro_body(screen, pluto)
            
            pygame.display.flip()
            clock.tick(FPS)
        pygame.quit()


    @staticmethod
    def show_body(screen:pygame.Surface, body: Body):
        rotated_rocket = pygame.transform.rotate(ROCKET_IMG, body.angle)
        new_rect = rotated_rocket.get_rect()
        body.center = (body.x, body.y)
        new_rect.center = body.center
        screen.blit(rotated_rocket, new_rect)
        pygame.draw.circle(screen, (0,0,0),new_rect.center,3)
    @staticmethod
    def draw_vectors(screen, body:Body):
        center = body.center
        for vec in body.vectors:
            color = (166, 9, 99)
            if vec.scalar == 9.8:
                color = (255,0,0)
            x2 = int(center[0] + (5+vec.scalar*2) * math.cos(Calc.degrees_to_radians(vec.angle)))
            y2 = int(center[1] - (5+vec.scalar*2) * math.sin(Calc.degrees_to_radians(vec.angle)))
            pygame.draw.line(screen, color, center, (x2,y2),2)
    @staticmethod
    def draw_astro_body(screen,astro_body):
        astro_body.draw(screen)


def main():
    Window.pygame_init()
    Window.load_assests()
    Window.run()


main()


            
