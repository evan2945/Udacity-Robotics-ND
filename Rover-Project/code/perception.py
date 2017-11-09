import numpy as np
import cv2

# Identify pixels above the threshold
# Threshold of RGB > 160 does a nice job of identifying ground pixels only
def color_thresh(img, rgb_thresh=(160, 160, 160), rgb_max_thresh=(255, 255, 255)):
    # Create an array of zeros same xy size as img, but single channel
    color_select = np.zeros_like(img[:,:,0])
    # Require that each pixel be above all three threshold values in RGB
    # above_thresh will now contain a boolean array with "True"
    # where threshold was met
    above_thresh = (img[:,:,0] > rgb_thresh[0]) \
                & (img[:,:,1] > rgb_thresh[1]) \
                & (img[:,:,2] > rgb_thresh[2]) \
                & (img[:, :, 0] < rgb_max_thresh[0]) \
                & (img[:, :, 1] < rgb_max_thresh[1]) \
                & (img[:, :, 2] < rgb_max_thresh[2])
    # Index the array of zeros with the boolean array and set to 1
    color_select[above_thresh] = 1
    # Return the binary image
    return color_select

# Define a function to convert from image coords to rover coords
def rover_coords(binary_img):
    # Identify nonzero pixels
    ypos, xpos = binary_img.nonzero()
    # Calculate pixel positions with reference to the rover position being at the
    # center bottom of the image.
    x_pixel = -(ypos - binary_img.shape[0]).astype(np.float)
    y_pixel = -(xpos - binary_img.shape[1]/2 ).astype(np.float)
    return x_pixel, y_pixel


# Define a function to convert to radial coords in rover space
def to_polar_coords(x_pixel, y_pixel):
    # Convert (x_pixel, y_pixel) to (distance, angle)
    # in polar coordinates in rover space
    # Calculate distance to each pixel
    dist = np.sqrt(x_pixel**2 + y_pixel**2)
    # Calculate angle away from vertical for each pixel
    angles = np.arctan2(y_pixel, x_pixel)
    return dist, angles

# Define a function to map rover space pixels to world space
def rotate_pix(xpix, ypix, yaw):
    # Convert yaw to radians
    yaw_rad = yaw * np.pi / 180
    xpix_rotated = (xpix * np.cos(yaw_rad)) - (ypix * np.sin(yaw_rad))

    ypix_rotated = (xpix * np.sin(yaw_rad)) + (ypix * np.cos(yaw_rad))
    # Return the result
    return xpix_rotated, ypix_rotated

def translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale):
    # Apply a scaling and a translation
    xpix_translated = (xpix_rot / scale) + xpos
    ypix_translated = (ypix_rot / scale) + ypos
    # Return the result
    return xpix_translated, ypix_translated


# Define a function to apply rotation and translation (and clipping)
# Once you define the two functions above this function should work
def pix_to_world(xpix, ypix, xpos, ypos, yaw, world_size, scale):
    # Apply rotation
    xpix_rot, ypix_rot = rotate_pix(xpix, ypix, yaw)
    # Apply translation
    xpix_tran, ypix_tran = translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale)
    # Perform rotation, translation and clipping all at once
    x_pix_world = np.clip(np.int_(xpix_tran), 0, world_size - 1)
    y_pix_world = np.clip(np.int_(ypix_tran), 0, world_size - 1)
    # Return the result
    return x_pix_world, y_pix_world

# Define a function to perform a perspective transform
def perspect_transform(img, src, dst):

    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))# keep same size as input image

    return warped


# Apply the above functions in succession and update the Rover state accordingly
def perception_step(Rover):

    # Make a note of our starting position
    if Rover.start is None:
        Rover.start = Rover.pos

    # 1) Define source and destination points for perspective transform
    source = np.float32([[14, 140],
                         [301, 140],
                         [200, 96],
                         [118, 96]])

    destination = np.float32([[Rover.img.shape[1] / 2 - 5, Rover.img.shape[0] - 6],
                              [Rover.img.shape[1] / 2 + 5, Rover.img.shape[0] - 6],
                              [Rover.img.shape[1] / 2 + 5, Rover.img.shape[0] - 10 - 6],
                              [Rover.img.shape[1] / 2 - 5, Rover.img.shape[0] - 10 - 6]])

    # 2) Apply perspective transform
    warped = perspect_transform(Rover.img, source, destination)

    # 3) Apply color threshold to identify navigable terrain/obstacles/rock samples
    nav_thresh = color_thresh(warped)
    obst_thresh = color_thresh(warped, rgb_thresh=(0, 0, 0), rgb_max_thresh=(160, 160, 160))
    rock_thresh = color_thresh(warped, rgb_thresh=(110, 90, 0), rgb_max_thresh=(255, 200, 50))

    # 4) Update Rover.vision_image (this will be displayed on left side of screen)
    Rover.vision_image[:, :, 0] = obst_thresh * 255
    Rover.vision_image[:, :, 1] = rock_thresh * 255
    Rover.vision_image[:, :, 2] = nav_thresh * 255

    # 5) Convert map image pixel values to rover-centric coords
    x_nav_pix, y_nav_pix = rover_coords(nav_thresh)
    x_obst_pix, y_obst_pix = rover_coords(obst_thresh)
    x_rock_pix, y_rock_pix = rover_coords(rock_thresh)

    # 6) Convert rover-centric pixel values to world coordinates
    x_world_nav, y_world_nav = pix_to_world(x_nav_pix, y_nav_pix, Rover.pos[0], Rover.pos[1], Rover.yaw, Rover.worldmap.shape[0], 10)
    x_world_obst, y_world_obst = pix_to_world(x_obst_pix, y_obst_pix, Rover.pos[0], Rover.pos[1], Rover.yaw, Rover.worldmap.shape[0], 10)
    x_world_rock, y_world_rock = pix_to_world(x_rock_pix, y_rock_pix, Rover.pos[0], Rover.pos[1], Rover.yaw, Rover.worldmap.shape[0], 10)

    # 7) Update Rover worldmap (to be displayed on right side of screen)
    # In this case, we want a higher fidelity, so only record to world map if we have minimum pitch and roll
    if (Rover.pitch < 0.5 or Rover.pitch > 359.5) and (Rover.roll < 0.5 or Rover.roll > 359.5):
        Rover.worldmap[y_world_obst, x_world_obst, 0] += 255
        Rover.worldmap[y_world_rock, x_world_rock, 1] += 255
        Rover.worldmap[y_world_nav, x_world_nav, 2] += 255


    # 8) Convert rover-centric pixel positions to polar coordinates
    # Update Rover pixel distances and angles
    dists, angles = to_polar_coords(x_nav_pix, y_nav_pix)
    Rover.nav_dists = dists
    Rover.nav_angles = angles

    # If we see a rock, we need to know its distance and angle from the rover for the sample pickup step
    rock_dist, rock_angle = to_polar_coords(x_rock_pix, y_rock_pix)

    # If we are within a certain distance from the sample, we need to switch modes. The main reason here for including
    # the distance is to keep the rover on the left side. I didn't want the rover to leave the left wall to grab a
    # sample on the right wall, as that would require more logic to have it go back to the left wall and continue. Since
    # I want to map as much of the world as possible, I'll be returning to that sample once I circle around anyways.
    if len(rock_dist) > 0 and np.mean(rock_dist) < 60:
        Rover.mode = 'rock_visible'
        Rover.nav_dists = rock_dist
        Rover.nav_angles = rock_angle
    # After pickup, switch back to forward mode
    else:
        if Rover.mode == 'rock_visible':
            Rover.mode = 'forward'
        dist, angles = to_polar_coords(x_nav_pix, y_nav_pix)
        Rover.nav_dists = dist
        Rover.nav_angles = angles


    return Rover
