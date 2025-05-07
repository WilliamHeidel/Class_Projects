import traceback
from collections import deque
from enum import Enum
import numpy as np
import torch
import torchvision
from torchvision.transforms import functional as F
from PIL import Image
import os
import math

from .models import load_model, Detector
from .models import CNNClassifier, load_model_CNN

GOALS = np.float32([[0, 64.5], [0, -64.5]])

LOST_STATUS_STEPS = 10
LOST_COOLDOWN_STEPS = 10
START_STEPS = 35
LAST_PUCK_DURATION = 4
MIN_SCORE = 0.2
MAX_DET = 15
MAX_DEV = 0.7
MIN_ANGLE = 20
MAX_ANGLE = 120
TARGET_SPEED = 15
STEER_YIELD = 15
DRIFT_THRESH = 0.7
TURN_CONE = 100


def get_device():
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')
    return device


device = get_device()


def calc_steering_angle(aim_point_x, aim_point_y):
    if aim_point_y < 0:
        steer = math.copysign(1, aim_point_x)
    elif aim_point_y == 0:
        steer = 0
    else:
        angle_radians = np.arctan(aim_point_x / aim_point_y)
        angle_degrees = np.degrees(angle_radians)
        normalized_angle = angle_degrees / 90
        steer = normalized_angle
    return steer


# Encapsulate anything we want to store about previous states (or the current state)
class History:
    def __init__(self):
        self.strategy: Strategy = Strategy.INIT
        self.x = 0
        self.y = 0
        self.pred = None
        self.puck_found = False


# The current strategy for our kart
class Strategy(Enum):
    # We think we see the puck and are trying to hit it
    PUCK = 1
    # We're within the first 32 frames of a new round
    INIT = 2
    # We were just reset to a new round
    RESET = 3
    # We are performing queued actions, like getting unstuck
    QUEUED = 4
    # We are circling looking for the puck
    CIRCLING = 5
    # We need to get unstuck
    UNSTICK = 6
    # We're backing up to try to find the puck again
    PUCK_BACK = 7


def export_images_new(player_num, image, action_taken, current_directory, team_number, current_history: History):
    img = image.unsqueeze(0)

    if team_number == 0:
        team = "red"
    elif team_number == 1:
        team = 'blue'
    else:
        team = 'unknown'

    relative_path = f"../controller_testing/new_detection_points_testing/detection_points_{team}_p{player_num}"
    absolute_path = os.path.join(current_directory, relative_path)
    os.makedirs(absolute_path, exist_ok=True)

    from datetime import datetime
    current_datetime = datetime.now()
    datetime_string = current_datetime.strftime('%Y%m%d%H%M%S%f')

    file_name = f"aim_point_{team}_p{player_num}_{datetime_string}.png"
    export_path = os.path.join(absolute_path, file_name)

    steer = 0
    if "steer" in action_taken:
        steer = action_taken["steer"]

    acceleration = 0
    if "acceleration" in action_taken:
        acceleration = action_taken["acceleration"]

    if current_history.pred is not None:
        pred_x = current_history.pred[1]
        pred_y = current_history.pred[2]
        pred_loc = f'({pred_x},{pred_y})'
        pred_center = np.array([[2 * (pred_x / 128) - 1, 2 * (pred_y / 96) - 1]])
        pred = torch.from_numpy(pred_center.astype(np.float32))
    else:
        pred = None
        pred_loc = 'No Puck'

    import matplotlib.pyplot as plt
    import torchvision.transforms.functional as TF
    fig, ax = plt.subplots(1, 1)
    ax.imshow(TF.to_pil_image(img[0].cpu()))
    WH2 = np.array([img.size(-1), img.size(-2)]) / 2
    if pred is not None:
        ax.add_artist(plt.Circle(WH2 * (pred[0].cpu().detach().numpy() + 1), 2, ec='r', fill=False, lw=1.5))
    ax.set_title(
        f'Pred: {pred_loc} | Det: {current_history.puck_found} | Steer: {round(steer, 3)}  | Acc: {round(acceleration, 3)} | Strat: {current_history.strategy.name} ')
    plt.axis('off')
    plt.savefig(export_path, bbox_inches='tight', pad_inches=0.1)
    plt.clf()
    plt.close()

    return None


class InternalPlayer:
    def __init__(self, player_num):
        self.player_num = player_num
        # Location during the previous act. We could queue these if we want a better trajectory
        self.last_location = None
        self.queued_steps = deque()
        self.history = deque()
        # Conceptually simpler for me :D
        self.direction_mapping = {
            (0, 1): "N",
            (1, 1): "NE",
            (1, 0): "E",
            (1, -1): "SE",
            (0, -1): "S",
            (-1, -1): "SW",
            (-1, 0): "W",
            (-1, 1): "NW",
        }
        # Enables clockwise or counterclockwise circling
        self.ordered_directions = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]

    # Fetch the previous history if available, set up a history for the current frame
    def previous_and_current_history(self, player_state):
        prev_history = self.history[-1] if len(self.history) > 0 else None
        current_history = History()
        x, y = self.current_xy(player_state)
        current_history.x = x
        current_history.y = y
        self.history.append(current_history)
        if len(self.history) > 5:
            self.history.popleft()
        return prev_history, current_history

    def was_game_reset(self, prev_hist: History, cur_hist: History):
        d_x = abs(cur_hist.x - prev_hist.x)
        d_y = abs(cur_hist.y - prev_hist.y)

        return True if d_x > 10 or d_y > 10 else False

    def add_multiple_reverse_steps(self, steer, count=4):
        for _ in range(count):
            self.queued_steps.append({"acceleration": 0, "brake": True, "steer": steer})

    # Logic for edge handling - get us away from the wall by queuing up reverse steps in the right direction
    def get_unstuck_if_near_edge(self, x, y, direction_facing):
        steer = None
        if x >= 35 and "E" in direction_facing:
            steer = 1 if y < 0 else -1
        elif x <= -35 and "W" in direction_facing:
            steer = 1 if y > 0 else -1
        elif y >= 65 and "N" in direction_facing:
            steer = -1 if x < 0 else 1
        elif y <= -65 and "S" in direction_facing:
            steer = -1 if x > 0 else 1

        if steer is not None:
            self.add_multiple_reverse_steps(steer)
            return {"acceleration": 0, "brake": True, "steer": steer}
        else:
            return None

    def circle_the_center(self, player_state):
        x, y = self.current_xy(player_state)
        direction_facing = self.direction_facing(player_state)

        unstick_action = self.get_unstuck_if_near_edge(x, y, direction_facing)

        if unstick_action is not None:
            return unstick_action, Strategy.UNSTICK

        dx = 0 - x
        dy = 0 - y

        # Normalize dx and dy
        dx = dx // abs(dx) if dx != 0 else 0
        dy = dy // abs(dy) if dy != 0 else 0

        # Find desired and current direction indices
        desired_direction = self.direction_mapping.get((dx, dy), "N")
        current_index = self.ordered_directions.index(direction_facing)
        desired_index = self.ordered_directions.index(desired_direction)

        circling_accel = 0.7

        # Adjust steering based on desired and current indices
        if current_index == desired_index:
            return {"acceleration": circling_accel, "steer": 0}, Strategy.CIRCLING
        elif (desired_index - current_index) % 8 <= 4:
            return {"acceleration": circling_accel, "steer": 1, "drift": True}, Strategy.CIRCLING
        else:
            return {"acceleration": circling_accel, "steer": -1, "drift": True}, Strategy.CIRCLING

    def direction_facing(self, player_state):
        x, y = self.current_xy(player_state)
        x_front = player_state['kart']['front'][0]
        y_front = player_state['kart']['front'][2]

        dx = x_front - x
        dy = y_front - y

        # Normalize dx and dy
        dx = dx // abs(dx) if dx != 0 else 0
        dy = dy // abs(dy) if dy != 0 else 0

        return self.direction_mapping.get((dx, dy), "N")

    def current_xy(self, player_state):
        return player_state['kart']['location'][0], player_state['kart']['location'][2]

    # If the puck is visible, head towards it
    def find_and_track_puck(self, pred_list):
        if len(pred_list) == 0:
            return None, None

        pred = pred_list[0][0]
        # pred_score = pred[0]

        # Don't steer after it if we're not confident
        # if pred_score < 0.5:
        #     return None, None

        pred_x = pred[1] - 63
        pred_y = 47 - pred[2]

        return {"acceleration": 0.5, "steer": calc_steering_angle(pred_x, pred_y)}, pred

    # Do we think we just passed the puck
    def previous_puck_likely(self, prev_hist) -> bool:
        if prev_hist.strategy == Strategy.PUCK:
            if prev_hist.pred is not None and prev_hist.pred[0] >= 0.5:
                # We were pretty sure about the last one
                return True

            # Two out of three previous
            if self.history[-2].strategy == Strategy.PUCK or self.history[-3].strategy == Strategy.PUCK:
                return True

        return False

    def get_last_puck_prediction(self):
        for i in range(-2, -5, -1):
            if self.history[i].pred is not None:
                return self.history[i].pred


class Team:
    agent_type = 'image'

    def __init__(self):
        """
          TODO: Load your agent here. Load network parameters, and other parts of our model
          We will call this function with default arguments only
        """
        self.step = 0
        self.kart = 'wilber'
        self.model = load_model().to(device)
        self.model.eval()
        self.cnn_model = load_model_CNN().to(device)
        self.cnn_model.eval()
        self.transform = torchvision.transforms.Compose([  # torchvision.transforms.Resize((96, 128)),
            torchvision.transforms.ToTensor()])
        # Trying to minimize what I touch for now
        self.internal_players = []
        self.sigmoid = torch.nn.Sigmoid()

    def new_match(self, team: int, num_players: int) -> list:
        """
        Let's start a new match. You're playing on a `team` with `num_players` and have the option of choosing your kart
        type (name) for each player.
        :param team: What team are you playing on RED=0 or BLUE=1
        :param num_players: How many players are there on your team
        :return: A list of kart names. Choose from 'adiumy', 'amanda', 'beastie', 'emule', 'gavroche', 'gnu', 'hexley',
                 'kiki', 'konqi', 'nolok', 'pidgin', 'puffy', 'sara_the_racer', 'sara_the_wizard', 'suzanne', 'tux',
                 'wilber', 'xue'. Default: 'tux'
        """
        try:
            for i in range(num_players):
                self.internal_players.append(InternalPlayer(i))
            self.team, self.num_players = team, num_players
            return ['tux'] * num_players
        except Exception as e:
            print(str(e))
            print(traceback.format_exc())
            raise e

    def is_puck_present(self, player_image):
        # Figure out puck presence
        o = self.cnn_model(player_image.unsqueeze(0))[0][0]
        return (self.sigmoid(o) > 0.5).item()

    def act_player(self, player: InternalPlayer, player_state, player_image, player_image_large):

        prev_hist: History
        current_hist: History
        prev_hist, current_hist = player.previous_and_current_history(player_state)

        if self.step < 34:
            # Game starts, drive straight so they can't score right out of the gate
            current_hist.strategy = Strategy.INIT
            return dict(acceleration=1, steer=0), current_hist

        # If we suddenly jumped a huge distancy that implies a goal was scored and we need to reset
        if player.was_game_reset(prev_hist, current_hist):
            #clear queue and history
            player.history.clear()
            player.queued_steps.clear()
            current_hist.strategy = Strategy.RESET
            self.step = 0
            return dict(acceleration=1, steer=0), current_hist

        # If we have queued steps, return the next one. Allows for multi step operations like escaping a wall
        if len(player.queued_steps) > 0:
            current_hist.strategy = Strategy.QUEUED
            return player.queued_steps.popleft(), current_hist

        puck_present = self.is_puck_present(player_image_large)
        current_hist.puck_found = puck_present

        if not puck_present and player.previous_puck_likely(prev_hist):
            # We're reasonably certain we saw the puck recently
            # Back up to try to find it
            previous_puck_x = player.get_last_puck_prediction()[1]
            steer = -1 if previous_puck_x < 0 else -1
            player.add_multiple_reverse_steps(steer, 4)
            current_hist.strategy = Strategy.PUCK_BACK
            return {"acceleration": 0, "brake": True, "steer": steer}, current_hist

        if puck_present:
            pred_list = self.model.detect(player_image)
            track_puck_action, pred = player.find_and_track_puck(pred_list)
            if track_puck_action is not None:
                current_hist.strategy = Strategy.PUCK
                current_hist.pred = pred
                return track_puck_action, current_hist

        action, desc = player.circle_the_center(player_state)
        current_hist.strategy = desc

        return action, current_hist

    def act(self, player_state, player_image):
        """
        This function is called once per timestep. You're given a list of player_states and images.

        DO NOT CALL any pystk functions here. It will crash your program on your grader.

        :param player_state: list[dict] describing the state of the players of this team. The state closely follows
                             the pystk.Player object <https://pystk.readthedocs.io/en/latest/state.html#pystk.Player>.
                             See HW5 for some inspiration on how to use the camera information.
                             camera:  Camera info for each player
                               - aspect:     Aspect ratio
                               - fov:        Field of view of the camera
                               - mode:       Most likely NORMAL (0)
                               - projection: float 4x4 projection matrix
                               - view:       float 4x4 view matrix
                             kart:  Information about the kart itself
                               - front:     float3 vector pointing to the front of the kart
                               - location:  float3 location of the kart
                               - rotation:  float4 (quaternion) describing the orientation of kart (use front instead)
                               - size:      float3 dimensions of the kart
                               - velocity:  float3 velocity of the kart in 3D

        :param player_image: list[np.array] showing the rendered image from the viewpoint of each kart. Use
                             player_state[i]['camera']['view'] and player_state[i]['camera']['projection'] to find out
                             from where the image was taken.

        :return: dict  The action to be taken as a dictionary. For example `dict(acceleration=1, steer=0.25)`.
                 acceleration: float 0..1
                 brake:        bool Brake will reverse if you do not accelerate (good for backing up)
                 drift:        bool (optional. unless you want to turn faster)
                 fire:         bool (optional. you can hit the puck with a projectile)
                 nitro:        bool (optional)
                 rescue:       bool (optional. no clue where you will end up though.)
                 steer:        float -1..1 steering angle
        """
        actions = []

        try:
            for i in range(self.num_players):
                image = F.to_tensor(Image.fromarray(player_image[i]).resize((128, 96))).to(device)
                image_large = F.to_tensor(Image.fromarray(player_image[i])).to(device)
                action, current_history = self.act_player(self.internal_players[i], player_state[i], image, image_large)
                actions.append(action)
                #export_images_new(i, image, action, os.getcwd(), self.team, current_history)
        except Exception as e:
            print(str(e))
            print(traceback.format_exc())
            raise e

        self.step += 1
        return actions