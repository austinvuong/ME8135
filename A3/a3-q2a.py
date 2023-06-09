import pygame
import numpy as np
from scipy.stats import multivariate_normal
from dataclasses import dataclass

# Config
SCALE_FACTOR = 480 * .04
WINDOW_WIDTH = 480
WINDOW_HEIGHT = 480

PREDICTIONS_PER_SECOND = 8
NUMBER_OF_PARTICLES = 200

STATE_DIM = 3

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

point_M = np.array([[10], [10]])

_world_offset = pygame.Vector2(WINDOW_WIDTH / 2 - point_M[0, 0] * SCALE_FACTOR, WINDOW_HEIGHT / 2 - (point_M[1, 0]+6) * SCALE_FACTOR) # offset to position the world on the screen

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
ground_truth = StateData(np.zeros((STATE_DIM, 1)), np.zeros((STATE_DIM, STATE_DIM)))
ground_truth.state[0] = point_M[0] 
ground_truth.state[1] = point_M[1] - 3
r = .1 # wheel radius (and also the speed in this case)
L = .3 # wheel base

# Motion model
transition_matrix = np.identity(STATE_DIM) # transition matrix
motion_matrix = np.identity(STATE_DIM) # motion matrix
def update_motion_matrix(theta, r, L): # updates the motion matrix with 
    motion_matrix[0, 0] = r * np.cos(theta)
    motion_matrix[1, 1] = r * np.sin(theta)
    motion_matrix[2, 2] = r/L
    return motion_matrix

# control vector
control_vector = np.zeros((STATE_DIM, 1))
def update_control_vector(u_r, u_l):
    control_vector[0] = 1/2 * (u_r + u_l) # u_omega
    control_vector[1] = 1/2 * (u_r + u_l) # u_omega
    control_vector[2] = u_r - u_l # u_psi
    return control_vector

# State Prediction (particle)
predicted = StateData(np.zeros((2, 1)), np.zeros((2, 2)))
particles = np.zeros((NUMBER_OF_PARTICLES, STATE_DIM, 1)) # particles[n] gets you the n'th particle as a column vector
for i in range(NUMBER_OF_PARTICLES):
    particles[i] = ground_truth.state
sd_w_omgea = .1
sd_w_psi = .01
prediction_covariance = np.array([
    [sd_w_omgea,             0,           0],
    [            0, sd_w_omgea,           0],
    [            0,             0, sd_w_psi]
]) # process covariance

# Update (i.e. measurement)
measurement = StateData(np.zeros((2, 1)), np.zeros((2, 2)))
measurement_matrix = np.zeros((2, STATE_DIM)) # observation matrix
measurement_matrix[0, 0] = 1
measurement_matrix[1, 1] = 2
measurement.covariance[0, 0] = .05 # sd_r_x
measurement.covariance[1, 1] = .075 # sd_r_y

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
            pygame.Vector2(*state_data.prev_state[0:2]) * SCALE_FACTOR + _world_offset,
            pygame.Vector2(*state_data.state[0:2]) * SCALE_FACTOR + _world_offset)

# Set the control inputs
update_control_vector(.11, .1)
control_vector *= 100

while _running:
    # poll for events
    # pygame.QUIT event means the user clicked X to close your window
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            _running = False
        elif event.type == PREDICTION_TIMER_EVENT:
            # print('prediction')
            vel_pred = motion_matrix @ control_vector # B @ u

            noise = np.random.multivariate_normal(np.zeros(STATE_DIM), prediction_covariance, size=NUMBER_OF_PARTICLES) 
            noise = noise.reshape(NUMBER_OF_PARTICLES, STATE_DIM, 1)

            particles += (vel_pred + noise) / PREDICTIONS_PER_SECOND

            # compute for the uncertainty elipise
            predicted.update_state(np.mean(particles, axis=0))
            predicted.covariance = np.cov(np.reshape(particles, (len(particles), -1)), rowvar=False)
            
        elif event.type == MEASUREMENT_TIMER_EVENT:
            # print('measurement')
            z = measurement_matrix @ ground_truth.state
            d = np.linalg.norm(z - particles[:, :2], axis=1) # distance between measurement and particles

            w = multivariate_normal.pdf(d) # unnormalized weights
            w /= np.sum(w, axis=0) # normalize
            cdf = np.cumsum(w)
            
            indexes = np.searchsorted(cdf, np.random.rand(NUMBER_OF_PARTICLES))
            particles = np.array([particles[i] for i in indexes])

            # for the uncertainty elipise 
            measurement.update_state(z)

    # Simulate ground truth
    d_state = motion_matrix @ control_vector # with noiseless motion
    d_state = np.random.multivariate_normal(d_state.flatten(), prediction_covariance).reshape(-1, 1) # with noisy motion
    ground_truth.update_state(ground_truth.state + d_state * _dt)

    update_motion_matrix(ground_truth.state[2], r, L)

    # print(particles[i])

    # Draw
    # Background
    _world_surface.fill('white')
    _uncertainty_surface.fill((0, 0, 0, 0))

    # Trajectory lines
    draw_trajectory_line(ground_truth, 'skyblue')
    draw_trajectory_line(predicted, 'lightgreen')
    draw_trajectory_line(measurement, 'darksalmon')

    # Particles
    for p in particles:
        if np.min(p[0:2].min() * SCALE_FACTOR + _world_offset) < 0: # skip these particles to avoid the issue draws smearing across the surface
            continue
        pygame.draw.circle(_world_surface, 'green', pygame.Vector2(*p[0:2]) * SCALE_FACTOR + _world_offset, 1, width=0)

    # Ground truth
    pygame.draw.circle(_world_surface, 'blue', pygame.Vector2(*ground_truth.state[0:2]) * SCALE_FACTOR + _world_offset, 3, width=0)

    # Point M
    pygame.draw.circle(_world_surface, 'black', pygame.Vector2(*point_M) * SCALE_FACTOR + _world_offset, 3, width=0)

    # Uncertainty
    # draw_covariance_ellipse(predicted, confidence=.95, color='green')
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