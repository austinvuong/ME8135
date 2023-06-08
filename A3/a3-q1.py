import pygame
import numpy as np
from numpy.linalg import inv
from scipy.stats import multivariate_normal
from dataclasses import dataclass

# Config
SCALE_FACTOR = 480*.5
WINDOW_WIDTH = 480
WINDOW_HEIGHT = 480

PREDICTIONS_PER_SECOND = 8
NUMBER_OF_PARTICLES = 100

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

_world_offset = pygame.Vector2(.5, .5) * SCALE_FACTOR # offset to position the world on the screen

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

# particles
particles = np.zeros((NUMBER_OF_PARTICLES, 2, 1)) # particles[n] gets you the n'th particle as a column vector

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
measurement = StateData(np.zeros((2, 1)), np.zeros((2, 2)))
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

    if (np.all(eigenvalues == 0)):
        # silent return
        # if all the eigenvalues are zero, there is no ellipse to draw
        return

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
            vel_pred = motion_matrix @ control_vector # B @ u

            noise = np.random.multivariate_normal(np.zeros(2), prediction_covariance, size=NUMBER_OF_PARTICLES) 
            noise = noise.reshape(NUMBER_OF_PARTICLES, 2, 1)

            particles += (vel_pred + noise) / PREDICTIONS_PER_SECOND

            # compute for the uncertainty elipises
            predicted.update_state(np.mean(particles, axis=0))
            predicted.covariance = np.cov(np.reshape(particles, (len(particles), -1)), rowvar=False)
            pass
        elif event.type == MEASUREMENT_TIMER_EVENT:
            # # print('measurement')
            # innovation =  - measurement_matrix @ predicted.state # C @ state - C @ state_pred
            # K = (predicted.covariance @ measurement_matrix.T
            #         @ inv(measurement_matrix @ predicted.covariance @ measurement_matrix.T + measureme5nt_covariance)
            #     ) # P_pred @ C^T @ (C @ P_pred @ C^T + R)^-1

            # # store for visualization
            # measurement.covariance = (np.identity(2) - K @ measurement_matrix) @ predicted.covariance # (I - K @ C) @ P_pred
            # measurement.update_state(predicted.state + K @ innovation) # state_pred + K @ innovation
            
            # # pass on the measurement
            # predicted.covariance = measurement.covariance
            # predicted.update_state(measurement.state)

            #WIP

            z = measurement_matrix @ ground_truth.state
            d = np.linalg.norm(z - particles, axis=1)

            w = multivariate_normal.pdf(d, None, measurement_covariance) # unnormalized weights #TODO #WIP the problem is here, it is inverting the weights
            w /= np.sum(w, axis=0) # normalize
            cdf = np.cumsum(w)
            
            indexes = np.searchsorted(cdf, np.random.rand(NUMBER_OF_PARTICLES))
            particles = [particles[i] for i in indexes]

            measurement.update_state(z)
            measurement.covariance = np.cov(np.reshape(particles, (len(particles), -1)), rowvar=False)
            pass

            #TODO figure out resampling

    # Simulate ground truth
    vel = motion_matrix @ control_vector # with noiseless motion
    vel = np.random.multivariate_normal(vel.flatten(), prediction_covariance).reshape(-1, 1) # with noisy motion
    ground_truth.update_state(ground_truth.state + vel * _dt)

    #TEMP visualzing test
    # particle += np.random.multivariate_normal(vel.flatten(), prediction_covariance).reshape(-1, 1) * _dt

    # Draw
    # Background
    _world_surface.fill('white')
    _uncertainty_surface.fill((0, 0, 0, 0))

    # Trajectory lines
    draw_trajectory_line(ground_truth, 'skyblue')
    draw_trajectory_line(predicted, 'lightgreen')
    draw_trajectory_line(measurement, 'darksalmon')

    # Ground truth
    pygame.draw.circle(_world_surface, 'blue', pygame.Vector2(*ground_truth.state) * SCALE_FACTOR + _world_offset, 3, width=0)

    # for p in particles: #TEMP
    #     pygame.draw.circle(_world_surface, 'magenta', pygame.Vector2(*p) * SCALE_FACTOR + _world_offset, 1, width=0)

    # Uncertainty
    draw_covariance_ellipse(predicted, confidence=.95, color='green')
    draw_covariance_ellipse(measurement, confidence=.95, color='red')

    # blit and flip
    _screen.blit(_world_surface, (0, 0))
    _screen.blit(_trajectory_surface, (0, 0))
    _screen.blit(_uncertainty_surface, (0, 0))
    pygame.display.flip()

    # dt is delta time in seconds since last frame, used for framerate-
    # independent physics.
    _dt = _clock.tick(60) / 1000

pygame.quit()