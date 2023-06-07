import pygame
import numpy as np
from numpy.linalg import inv
from dataclasses import dataclass

# Config
SCALE_FACTOR = 480
WINDOW_WIDTH = 480
WINDOW_HEIGHT = 480

PREDICTIONS_PER_SECOND = 8

np.random.seed(42) # fix for reproducution

# pygame setup
pygame.init()

_screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT)) # the window # TEMP during recordings add , pygame.NOFRAME
_world_surface = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA) # the main drawing surface
_trajectory_surface = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA) # for drawing trajectory of the robot
_uncertainty_surface = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA) # for drawing uncertainty ellipses

_clock = pygame.time.Clock()
_running = True
_dt = 0

_world_offset = pygame.Vector2(0, 0) # offset to position the world on the screen

@dataclass
class StateData:
    state: np.ndarray
    prev_state: np.ndarray  # the previous state (singular). For trajectory visualization
    covariance: np.ndarray

    def __init__(self, state: np.ndarray, covariance: np.ndarray):
        self.state = state
        self.prev_state = state.copy()
        self.covariance = covariance

    def update_state(self, new_state: np.ndarray):
        self.prev_state = self.state
        self.state = new_state

# robot
ground_truth = StateData(np.zeros((2, 1)), np.zeros((2, 2)))
r = .1 # wheel radius (and also the speed in this case)

# Motion model
transition_matrix = np.identity(2) # transition matrix
motion_matrix = r/2 * np.identity(2) # motion matrix
u_r = u_l = 1 # chosen control inputs
control_vector = np.array([[u_r + u_l], [u_r + u_l]]) # control vector

# Prediction (i.e. odometry)
predicted = StateData(np.zeros((2, 1)), np.zeros((2, 2)))
sd_w_x = .1
sd_w_y = .15
prediction_covariance = np.array([[sd_w_x, 0], [0, sd_w_y]]) # process covariance

# Update (i.e. measurement)
measurment = StateData(np.zeros((2, 1)), np.zeros((2, 2)))
measurement_matrix = np.array([[1, 0], [0, 2]]) # observation matrix
sd_r_x = .05
sd_r_y = .075
measurement_covariance = np.array([[sd_r_x, 0], [0, sd_r_y]]) # measurement covariance

# Timers
prediction_timer_delay = round(1000/PREDICTIONS_PER_SECOND)
PREDICTION_TIMER_EVENT = pygame.USEREVENT + 0
pygame.time.set_timer(PREDICTION_TIMER_EVENT, prediction_timer_delay)

measurement_timer_delay = 1000
MEASUREMENT_TIMER_EVENT = pygame.USEREVENT + 1
pygame.time.set_timer(MEASUREMENT_TIMER_EVENT, measurement_timer_delay)

# Taken from https://stackoverflow.com/questions/65767785/how-to-draw-a-rotated-ellipse-using-pygame
def draw_ellipse_angle(surface, color, rect, angle, width=0):
    target_rect = pygame.Rect(rect)
    shape_surf = pygame.Surface(target_rect.size, pygame.SRCALPHA)
    pygame.draw.ellipse(shape_surf, color, (0, 0, *target_rect.size), width)
    rotated_surf = pygame.transform.rotate(shape_surf, angle)
    surface.blit(rotated_surf, rotated_surf.get_rect(center = target_rect.center))

# assumes that state_data.state starts with meaningful x and y
def draw_covariance_ellipse(state_data: StateData, confidence=.95, color='magenta'):
    eigenvalues, eigenvectors = np.linalg.eig(state_data.covariance)

    s = np.sqrt(-2 * np.log(1 - confidence)) # simple 
    major = 2 * np.sqrt(np.max(eigenvalues) * SCALE_FACTOR) * s
    minor = 2 * np.sqrt(np.min(eigenvalues) * SCALE_FACTOR) * s
    
    max_eigenvalue_index = np.argmax(eigenvalues)
    angle = -np.degrees(np.arctan2(eigenvectors[max_eigenvalue_index, 1], eigenvectors[max_eigenvalue_index, 0]))
    
    rect = (state_data.state[0, 0] * SCALE_FACTOR - major/2 + _world_offset.x, 
            state_data.state[1, 0] * SCALE_FACTOR - minor/2 + _world_offset.y, 
            major, minor)
    draw_ellipse_angle(_uncertainty_surface, color, rect, angle, 2)

def draw_trajectory_line(state_data: StateData, color='magenta'):
        pygame.draw.line(_trajectory_surface, color, 
            pygame.Vector2(*state_data.prev_state) * SCALE_FACTOR + _world_offset,
            pygame.Vector2(*state_data.state) * SCALE_FACTOR + _world_offset)

while _running:
    # poll for events
    # pygame.QUIT event means the user clicked X to close your window
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            _running = False
        elif event.type == PREDICTION_TIMER_EVENT:
            # print('prediction')
            vel_pred = motion_matrix @ control_vector / PREDICTIONS_PER_SECOND # B @ u * timing

            # store for visualization
            predicted.covariance = transition_matrix @ predicted.covariance @ transition_matrix.T + prediction_covariance # P_pred = A @ P_pred @ A^T + Q
            predicted.update_state(transition_matrix @ predicted.state + vel_pred) # state_pred = A @ state_pred + B @ u * timing

        elif event.type == MEASUREMENT_TIMER_EVENT:
            # print('measurement')
            innovation = measurement_matrix @ ground_truth.state - measurement_matrix @ predicted.state # C @ state - C @ state_pred
            K = (predicted.covariance @ measurement_matrix.T
                    @ inv(measurement_matrix @ predicted.covariance @ measurement_matrix.T + measurement_covariance)
                ) # P_pred @ C^T @ (C @ P_pred @ C^T + R)^-1

            # store for visualization
            measurment.covariance = (np.identity(2) - K @ measurement_matrix) @ predicted.covariance # (I - K @ C) @ P_pred
            measurment.update_state(predicted.state + K @ innovation) # state_pred + K @ innovation
            
            # pass on the measurement
            predicted.covariance = measurment.covariance
            predicted.update_state(measurment.state)

    # Simulate ground truth
    vel = np.random.multivariate_normal((motion_matrix @ control_vector).flatten(), prediction_covariance).reshape(-1, 1) # with noisy motion
    # vel = motion_matrix @ control_vector # with noiseless motion
    ground_truth.update_state(ground_truth.state + vel * _dt)

    # Draw
    # Background
    _world_surface.fill('white')
    _uncertainty_surface.fill((0, 0, 0, 0))

    # Trajectory lines
    draw_trajectory_line(ground_truth, 'skyblue')
    draw_trajectory_line(predicted, 'lightgreen')
    draw_trajectory_line(measurment, 'darksalmon')

    # Ground truth
    pygame.draw.circle(_world_surface, 'blue', pygame.Vector2(*ground_truth.state) * SCALE_FACTOR + _world_offset, 3, width=0)

    # Uncertainty
    draw_covariance_ellipse(predicted, confidence=.95, color='green')
    draw_covariance_ellipse(measurment, confidence=.95, color='red')

    # blit and flip
    _screen.blit(_world_surface, (0, 0))
    _screen.blit(_trajectory_surface, (0, 0))
    _screen.blit(_uncertainty_surface, (0, 0))
    pygame.display.flip()

    # dt is delta time in seconds since last frame, used for framerate-
    # independent physics.
    _dt = _clock.tick(60) / 1000

pygame.quit()