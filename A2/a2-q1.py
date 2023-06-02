import pygame
import numpy as np
from numpy.linalg import inv

# Config
SCALE_FACTOR = 480
WINDOW_WIDTH = 480
WINDOW_HEIGHT = 480

PREDICTIONS_PER_SECOND = 8

np.random.seed(42) # fix for reproducution

# pygame setup
pygame.init()

screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT)) # the window
world_surface = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA) # the main drawing surface
trajectory_surface = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA) # for drawing trajectory of the robot
uncertainty_surface = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA) # for drawing uncertainty ellipses

clock = pygame.time.Clock()
running = True
dt = 0

offset = pygame.Vector2(0, 0) # offset to position the world on the screen

# robot
pos = np.zeros((2, 1))
last_position = pos.copy() # for trajectory visualization
r = .1 # wheel radius (and also the speed in this case)

# Motion model
A = np.identity(2) # transition matrix
B = r/2 * np.identity(2) # motion matrix
u_r = u_l = 1 # chosen control inputs
u = np.array([[u_r + u_l], [u_r + u_l]]) # control vector

# Prediction (i.e. odometry)
pos_pred = np.zeros((2, 1))
last_pos_pred = pos_pred.copy() # for trajectory visualization
P_pred = np.zeros((2, 2)) # prediction covariance
sd_w_x = .1
sd_w_y = .15
R = np.array([[sd_w_x, 0], [0, sd_w_y]]) # process covariance

# Update (i.e. measurement)
pos_corrected = np.zeros((2, 1))
last_pos_corrected = pos_corrected.copy() # for trajectory visualization
P_corrected = np.zeros((2, 2)) # measurement covariance
C = np.array([[1, 0], [0, 2]]) # observation matrix
sd_r_x = .05
sd_r_y = .075
Q = np.array([[sd_r_x, 0], [0, sd_r_y]]) # measurement covariance

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

def draw_covariance_ellipse(pos, P, confidence=.95, color='purple'):
    eigenvalues, eigenvectors = np.linalg.eig(P)

    s = np.sqrt(-2 * np.log(1 - confidence)) # simple 
    major = 2 * np.sqrt(np.max(eigenvalues) * SCALE_FACTOR) * s
    minor = 2 * np.sqrt(np.min(eigenvalues) * SCALE_FACTOR) * s
    
    max_eigenvalue_index = np.argmax(eigenvalues)
    angle = -np.degrees(np.arctan2(eigenvectors[max_eigenvalue_index, 1], eigenvectors[max_eigenvalue_index, 0]))
    
    rect = (pos[0, 0] * SCALE_FACTOR - major/2 + offset.x, 
            pos[1, 0] * SCALE_FACTOR - minor/2 + offset.y, 
            major, minor)
    draw_ellipse_angle(uncertainty_surface, color, rect, angle, 2)

while running:
    # poll for events
    # pygame.QUIT event means the user clicked X to close your window
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == PREDICTION_TIMER_EVENT:
            # print('prediction')
            vel_pred = B @ u / PREDICTIONS_PER_SECOND

            # store for visualization
            P_pred = A @ P_pred @ A.T + R # (3.120a) modified for multiple predictions
            pos_pred = A @ pos_pred + vel_pred # (3.120b) modified for multiple predictions

        elif event.type == MEASUREMENT_TIMER_EVENT:
            # print('measurement')
            z = C @ pos
            innovation = (z - C @ pos_pred)
            K = P_pred @ C.T @ inv(C @ P_pred @ C.T + Q)

            # store for visualization
            P_corrected = (np.identity(2) - K @ C) @ P_pred
            pos_corrected = pos_pred + K @ innovation
            
            # pass on the measurement
            P_pred = P_corrected
            pos_pred = pos_corrected

    # Simulate ground truth
    vel = np.random.multivariate_normal((B @ u).flatten(), R).reshape(-1, 1) # with noisy motion
    # vel = B @ u # with noiseless motion
    pos += vel * dt

    # Draw
    # Background
    world_surface.fill('white')
    uncertainty_surface.fill((0, 0, 0, 0))

    # Trajectory lines
    pygame.draw.line(trajectory_surface, 'skyblue', 
        pygame.Vector2(*last_position) * SCALE_FACTOR + offset,
        pygame.Vector2(*pos) * SCALE_FACTOR + offset)
    last_position = pos.copy()

    pygame.draw.line(trajectory_surface, 'lightgreen', 
        pygame.Vector2(*last_pos_pred) * SCALE_FACTOR + offset,
        pygame.Vector2(*pos_pred) * SCALE_FACTOR + offset)
    last_pos_pred = pos_pred.copy()

    pygame.draw.line(trajectory_surface, 'darksalmon', 
        pygame.Vector2(*last_pos_corrected) * SCALE_FACTOR + offset,
        pygame.Vector2(*pos_corrected) * SCALE_FACTOR + offset)
    last_pos_corrected = pos_corrected.copy()

    # Ground truth
    pygame.draw.circle(world_surface, 'blue', pygame.Vector2(*pos) * SCALE_FACTOR + offset, 3, width=0)

    # Uncertainty
    draw_covariance_ellipse(pos_pred, P_pred, confidence=.95, color='green')
    draw_covariance_ellipse(pos_corrected, P_corrected, confidence=.95, color='red')

    # blit and flip
    screen.blit(world_surface, (0, 0))
    screen.blit(trajectory_surface, (0, 0))
    screen.blit(uncertainty_surface, (0, 0))
    pygame.display.flip()

    # dt is delta time in seconds since last frame, used for framerate-
    # independent physics.
    dt = clock.tick(60) / 1000

pygame.quit()