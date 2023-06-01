import pygame
import numpy as np
from numpy.linalg import inv

SCALE_FACTOR = 480 * .9
PREDICTIONS_PER_SECOND = 8
WINDOW_WIDTH = 480
WINDOW_HEIGHT = 480

np.random.seed(42)

# pygame setup
pygame.init()

screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT)) # the window
world_surface = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA) # the main drawing surface
trajectory_surface = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA) # for drawing trajectory of the robot
uncertainty_surface = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA) # for drawing uncertainty ellipses
last_position = np.zeros((2, 1)) # track the last position for drawing the trajectory

clock = pygame.time.Clock()
running = True
dt = 0

point_M = np.array([[10], [10]])

offset = pygame.Vector2(WINDOW_WIDTH / 2 - point_M[0, 0] * SCALE_FACTOR, WINDOW_HEIGHT / 2 - point_M[1, 0] * SCALE_FACTOR) # offset to position the world on the screen

# robot
state = np.zeros((3, 1))
state[0] = point_M[0]
state[1] = point_M[1] - .3
r = .1 # wheel radius (and also the speed in this case)
L = .3 # wheel base

A = np.identity(3) # transition matrix

# motion matrix
B = np.identity(3)
def update_B(theta, r, L):
    B[0, 0] = r * np.cos(theta)
    B[1, 1] = r * np.sin(theta)
    B[2, 2] = r/L
    return B

# control vector
u = np.zeros((3, 1))
def update_u(u_r, u_l):
    u[0] = 1/2 * (u_r + u_l) # u_omega
    u[1] = 1/2 * (u_r + u_l) # u_omega
    u[2] = u_r - u_l # u_psi
    return u

# State Prediction (odometry)
state_pred = state.copy()
P_pred = np.zeros((3, 3))
P = np.zeros((3, 3)) # prediction (position) covariance
sd_w_omgea = .1
sd_w_psi = .01
Q = np.array([
    [sd_w_omgea**2,             0,           0],
    [            0, sd_w_omgea**2,           0],
    [            0,             0, sd_w_psi**2]
]) # process covariance

# State Update (measurement) (from A2a)
state_corrected = np.zeros((2, 1))
P_corrected = np.zeros((3, 3))
C = np.zeros((2, 3)) # observation matrix
C[0, 0] = 1
C[1, 1] = 2
sd_r_x = .05
sd_r_y = .075
R = np.array([[sd_r_x**2, 0], [0, sd_r_y**2]]) # measurement covariance

# State Update (landmark) (from A2b)
state_landmark_corrected = np.zeros((2, 1))
P_landmark_corrected = np.zeros((3, 3))
D = np.zeros((2, 3)) # landmark matrix
def update_D(x, y):
    # d1 = rho = (range) = sqrt(X^2 + Y^2)
    # d2 = phi = (bearing) = arctan(Y/X)
    X = point_M[0] - x
    Y = point_M[1] - y
    X2Y2 = X ** 2 + Y ** 2

    # The Jacobian
    D[0, 0] = Y / np.sqrt(X2Y2) # del d1 / del x
    D[0, 1] = X / np.sqrt(X2Y2) # del d1 / del y
    D[1, 1] = Y / X2Y2 # del d2 / del x
    D[1, 1] = X / X2Y2 # del d2 / del y
    return D

sd_range = .1
sd_bearing = .01
S = np.array([[sd_range**2, 0], [0, sd_bearing**2]]) # landmark covariance

# Timers
prediction_timer_delay = round(1000/PREDICTIONS_PER_SECOND)
PREDICTION_TIMER_EVENT = pygame.USEREVENT + 0
pygame.time.set_timer(PREDICTION_TIMER_EVENT, prediction_timer_delay)

measurement_timer_delay = 1000
MEASUREMENT_TIMER_EVENT = pygame.USEREVENT + 1
pygame.time.set_timer(MEASUREMENT_TIMER_EVENT, measurement_timer_delay)

landmark_timer_delay = 700 # chossen avoid overlapping the measurement timer
LANDMARK_TIMER_EVENT = pygame.USEREVENT + 2
pygame.time.set_timer(LANDMARK_TIMER_EVENT, landmark_timer_delay)

# Taken from https://stackoverflow.com/questions/65767785/how-to-draw-a-rotated-ellipse-using-pygame
def draw_ellipse_angle(surface, color, rect, angle, width=0):
    target_rect = pygame.Rect(rect)
    shape_surf = pygame.Surface(target_rect.size, pygame.SRCALPHA)
    pygame.draw.ellipse(shape_surf, color, (0, 0, *target_rect.size), width)
    rotated_surf = pygame.transform.rotate(shape_surf, angle)
    surface.blit(rotated_surf, rotated_surf.get_rect(center = target_rect.center))

def draw_covariance_ellipse(state, P, confidence=.95, color='purple'):
    eigenvalues, eigenvectors = np.linalg.eig(P)

    if (np.all(eigenvalues == 0)):
        # silent return
        # if all the eigenvalues are zero, there is no ellipse to draw
        return

    s = np.sqrt(-2 * np.log(1 - confidence)) # simple 
    major = 2 * np.sqrt(np.max(eigenvalues) * SCALE_FACTOR) * s
    minor = 2 * np.sqrt(np.min(eigenvalues[eigenvalues > 0.00001]) * SCALE_FACTOR) * s

    max_eigenvalue_index = np.argmax(eigenvalues)
    angle = -np.degrees(np.arctan2(eigenvectors[max_eigenvalue_index, 1], eigenvectors[max_eigenvalue_index, 0]))

    rect = (state[0, 0] * SCALE_FACTOR - major/2 + offset.x, 
            state[1, 0] * SCALE_FACTOR - minor/2 + offset.y, 
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
            P_pred = A @ P_pred @ A.T + Q # (3.120a) modified for multiple predictions
            state_pred = A @ state_pred + vel_pred # (3.120b) modified for multiple predictions

        elif event.type == MEASUREMENT_TIMER_EVENT:
            # print('measurement')
            y = C @ state

            K = P_pred @ C.T @ inv(C @ P_pred @ C.T + R)

            innovation = (y - C @ state_pred)
            # print(f'K: {K}')
            # print(f'innovation: {innovation}')
            
            P_corrected = (np.identity(3) - K @ C) @ P_pred
            state_corrected = state_pred + K @ innovation
            
            P_pred = P_corrected
            state_pred = state_corrected
        elif event.type == LANDMARK_TIMER_EVENT:
            # print('measurement')
            update_D(*state[0:2])
            y = D @ state

            K = P_pred @ D.T @ inv(D @ P_pred @ D.T + S)

            innovation = (y - D @ state_pred)
            # print(f'K: {K}')
            # print(f'innovation: {innovation}')
            
            P_landmark_corrected = (np.identity(3) - K @ D) @ P_pred
            state_landmark_corrected = state_pred + K @ innovation
            
            P_pred = P_landmark_corrected
            state_pred = state_landmark_corrected

    # Simulate ground truth
    d_state = np.random.multivariate_normal((B @ u).flatten(), Q).reshape(-1, 1) # with noisy motion
    # d_state = B @ u # with noiseless motion
    # d_state[2] = np.radians(d_state[2])
    state += d_state * dt

    update_u(.7, .3)
    u *= 5 #TEMP to speed up tests
    update_B(state[2], r, L)

    # Draw
    # Background
    world_surface.fill('white')
    uncertainty_surface.fill((0, 0, 0, 0))

    # Trajectory line
    pygame.draw.line(trajectory_surface, 'lightblue', 
        pygame.Vector2(*last_position) * SCALE_FACTOR + offset,
        pygame.Vector2(*state[0:2]) * SCALE_FACTOR + offset)
    last_position = state[0:2].copy()

    # Ground truth
    pygame.draw.circle(world_surface, 'blue', pygame.Vector2(*state[0:2]) * SCALE_FACTOR + offset, 3, width=0)

    # point M
    pygame.draw.circle(world_surface, 'black', pygame.Vector2(*point_M) * SCALE_FACTOR + offset, 3, width=0)

    # Uncertainty
    draw_covariance_ellipse(state_pred, P_pred, confidence=.95, color='green')
    draw_covariance_ellipse(state_pred, P_pred, confidence=.5, color='lightgreen')
    draw_covariance_ellipse(state_corrected, P_corrected, confidence=.95, color='red')
    draw_covariance_ellipse(state_landmark_corrected, P_landmark_corrected, confidence=.95, color='orange')
    
    # blit and flip
    screen.blit(world_surface, (0, 0))
    screen.blit(trajectory_surface, (0, 0))
    screen.blit(uncertainty_surface, (0, 0))
    pygame.display.flip()

    # dt is delta time in seconds since last frame, used for framerate-
    # independent physics.
    dt = clock.tick(60) / 1000

pygame.quit()