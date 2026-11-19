# my_team.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import random
import contest.util as util

from contest.capture_agents import CaptureAgent
from contest.game import Directions
from contest.util import nearest_point


#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='OffensiveReflexAgent', second='DefensiveReflexAgent', num_training=0):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.

    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --red_opts and --blue_opts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """
    return [eval(first)(first_index), eval(second)(second_index)]


##########
# Agents #
##########

class ReflexCaptureAgent(CaptureAgent):
    """
    A base class for reflex agents that choose score-maximizing actions
    """

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.start = None

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)
  
        # afegim quina es la frontera del nostre camp ja que ho necessitarem
        walls = game_state.get_walls()
        mig = walls.width // 2
        if self.red:
            limit_x = mig - 1
        else:
            limit_x = mig
        self.boundary = []
        for y in range(walls.height):
            if not walls[limit_x][y]:
                self.boundary.append((limit_x, y))



    def choose_action(self, game_state):
        """
        Picks among the actions with the highest Q(s,a).
        """
        actions = game_state.get_legal_actions(self.index)

        # You can profile your evaluation time by uncommenting these lines
        # start = time.time()
        values = [self.evaluate(game_state, a) for a in actions]
        # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]

        food_left = len(self.get_food(game_state).as_list())

        if food_left <= 2:
            best_dist = 9999
            best_action = None
            for action in actions:
                successor = self.get_successor(game_state, action)
                pos2 = successor.get_agent_position(self.index)
                dist = self.get_maze_distance(self.start, pos2)
                if dist < best_dist:
                    best_action = action
                    best_dist = dist
            return best_action

        return random.choice(best_actions)

    def get_successor(self, game_state, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearest_point(pos):
            # Only half a grid position was covered
            return successor.generate_successor(self.index, action)
        else:
            return successor

    def evaluate(self, game_state, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.get_features(game_state, action)
        weights = self.get_weights(game_state, action)
        return features * weights

    # funció per saber la distancia a la que estam d'arribar al nostre camp
    def distance_to_home(self, pos):
        min_dist = 9999
        for cell in self.boundary:
            d = self.get_maze_distance(pos, cell)
            if d < min_dist:
                min_dist = d
        return min_dist

    # funció per saber si tenim rivals aprop
    def visible_enemies(self, game_state):
        enemies = []
        for i in self.get_opponents(game_state):
            state = game_state.get_agent_state(i)
            if state.get_position() is not None: # si estan enfora el get_position() retorna None
                enemies.append(state)
        return enemies
    
    def get_features(self, game_state, action):
        """
        Returns a counter of features for the state
        """
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        features['successor_score'] = self.get_score(successor)
        return features

    def get_weights(self, game_state, action):
        """
        Normally, weights do not depend on the game state.  They can be either
        a counter or a dictionary.
        """
        return {'successor_score': 1.0}


class OffensiveReflexAgent(ReflexCaptureAgent):
    """
  A reflex agent that seeks food. This is an agent
  we give you to get an idea of what an offensive agent might look like,
  but it is by no means the best or only way to build an offensive agent.
  """

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        food_list = self.get_food(successor).as_list()
        features['successor_score'] = -len(food_list)  # self.getScore(successor)

        # Compute distance to the nearest food

        if len(food_list) > 0:  # This should always be True,  but better safe than sorry
            my_pos = successor.get_agent_state(self.index).get_position()
            min_distance = min([self.get_maze_distance(my_pos, food) for food in food_list])
            features['distance_to_food'] = min_distance
        
        # informació de després de fer l'acció
        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()
        carrying = my_state.num_carrying   # peces de menjar que duim

        # si l'score puja és que travessam la frontera (ho incentivam)
        features['punts'] = self.get_score(successor)

        # miram quin enemic visible que sigui perillós (no estigui espantat o li quedi poc) tenim més a prop
        ghost_dist = 9999
        for enemy in self.visible_enemies(successor):
            if not enemy.is_pacman and enemy.scared_timer <= 2:
                d = self.get_maze_distance(my_pos, enemy.get_position())
                if d < ghost_dist:
                    ghost_dist = d

        # tornam a casa si es cumpleix alguna d'aquestes condicions
        time_left = game_state.data.timeleft
        go_home = (carrying >= 5
                   or len(food_list) <= 2
                   or (carrying > 0 and time_left < 4 * (self.distance_to_home(my_pos) + 10))
                   or (carrying > 0 and ghost_dist <= 5))
        if go_home:
            # ja no buscam menjar i anem cap al nostre camp
            features['distance_to_food'] = 0
            features['dist_camp'] = self.distance_to_home(my_pos)

        # penalitzam acostar-mos als fantasmes
        if my_state.is_pacman and ghost_dist <= 5:
            features['dist_ghost'] = 1.0 / max(ghost_dist, 1)
            if ghost_dist <= 1:
                features['ghost_aferrat'] = 1

        # si després de l'acció som a la posició inicial i abans no hi erem es que ens han menjat
        if my_pos == self.start and game_state.get_agent_position(self.index) != self.start:
            features['mort'] = 1

        # si tenim un fantasma a prop i hi ha alguna capsula a prop anirem a buscar la per poder menjar o fugir del fantasma
        capsules = self.get_capsules(successor)
        if len(capsules) > 0 and ghost_dist <= 5:
            min_cap = 9999
            for cap in capsules:
                d = self.get_maze_distance(my_pos, cap)
                if d < min_cap:
                    min_cap = d
            features['dist_capsula'] = min_cap

        # penalitzem no mourer-se i tornar cap enrere perquè l'agent no quedi en bucle 
        if action == Directions.STOP:
            features['stop'] = 1
        anar_enrere = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == anar_enrere:
            features['enrere'] = 1

        return features

    def get_weights(self, game_state, action):
        return {'successor_score': 100, 'distance_to_food': -2,
                # pesos de les features que hem afegit
                'punts': 100, 'dist_camp': -5, 'dist_ghost': -50,
                'ghost_aferrat': -1000, 'mort': -1000, 'dist_capsula': -3,
                'stop': -100, 'enrere': -2}


class DefensiveReflexAgent(ReflexCaptureAgent):
    """
    A reflex agent that keeps its side Pacman-free. Again,
    this is to give you an idea of what a defensive agent
    could be like.  It is not the best or only way to make
    such an agent.
    """

    # guardam alla on mengen per darrera vegada
    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.last_eaten = None

    def get_features(self, game_state, action):
        features = util.Counter()
    
        # si mos han menjat significa que hi ha un pacman a prop d'allà
        previous = self.get_previous_observation()
        if previous is not None: # si no te previa es que es el primer torn i no hi ha res amb que comparar
            menjar_abans = set(self.get_food_you_are_defending(previous).as_list())
            menjar_despres = set(self.get_food_you_are_defending(game_state).as_list())
            menjat = menjar_abans - menjar_despres
            if len(menjat) > 0:
                self.last_eaten = list(menjat)[0]

        successor = self.get_successor(game_state, action)

        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()

        # Computes whether we're on defense (1) or offense (0)
        features['on_defense'] = 1
        if my_state.is_pacman: features['on_defense'] = 0

        # Computes distance to invaders we can see
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        features['num_invaders'] = len(invaders)
        if len(invaders) > 0:
            dists = [self.get_maze_distance(my_pos, a.get_position()) for a in invaders]
            features['invader_distance'] = min(dists)

            # si estam scared ens poden menjar per tant ja no ens apropam als pacman
            # ens mantenim a 2 caselles per menjar-lo quan ja poguem
            if my_state.scared_timer > 0:
                features['invader_distance'] = 0
                features['dist_quan_scared'] = abs(min(dists) - 2)
        else:
            # si no veim invasors anem cap a on ens han agafat menjar i si no han menjat anem cap al centre del taulell
            if self.last_eaten is not None and my_pos != self.last_eaten:
                features['dist_objectiu_vigilar'] = self.get_maze_distance(my_pos, self.last_eaten)
            else:
                if my_pos == self.last_eaten:
                    self.last_eaten = None   # quan ja arribam allà on haviem d'anar ja no necessitam anar cap allà
                walls = game_state.get_walls()
                centre = self.boundary[0]
                best = 9999
                for cell in self.boundary:
                    d = abs(cell[1] - walls.height // 2)
                    if d < best:
                        best = d
                        centre = cell
                features['dist_objectiu_vigilar'] = self.get_maze_distance(my_pos, centre)

        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1

        return features

    def get_weights(self, game_state, action):
                return {'num_invaders': -1000, 'on_defense': 100, 'invader_distance': -10, 'stop': -100, 'reverse': -2,
                # pesos de les features afegides
                'dist_quan_scared': -10, 'dist_objectiu_vigilar': -2}
