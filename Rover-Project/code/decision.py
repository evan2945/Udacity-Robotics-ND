import numpy as np
import time


# This is where you can build a decision tree for determining throttle, brake and steer
# commands based on the output of the perception_step() function
def decision_step(Rover):

    # If we are stopped near a sample, send the pickup signal
    if Rover.near_sample and Rover.vel == 0 and not Rover.picking_up:
        Rover.send_pickup = True
    # If we can see a sample and are in rock_visible mode, we need to manage our throttle and braking depending
    # on how close we are to it
    elif Rover.mode == 'rock_visible':
        mean_dist = np.mean(Rover.nav_dists)
        if (mean_dist < Rover.stop_forward) and (mean_dist > 15) and Rover.vel != 0:
            Rover.throttle = 0.1
        elif mean_dist < 15 and Rover.near_sample:
            Rover.brake = Rover.brake_set
        elif mean_dist < 15 and Rover.brake > 5 and Rover.vel == 0 and not Rover.near_sample:
            Rover.brake = 0
            Rover.throttle = 0.1
        else:
            Rover.throttle = Rover.throttle_set
        # Set steering to average angle clipped to the range +/- 15
        Rover.steer = np.clip(np.mean(Rover.nav_angles * 180 / np.pi), -15, 15)

    # Example:
    # Check if we have vision data to make decisions with
    elif Rover.nav_angles is not None:
        # Check for Rover.mode status
        if Rover.mode == 'forward':
            # This keeps track if our rover hasn't moved in a certain amount of time. If it hasn't, switch to stuck mode
            if Rover.vel < 0.25:
                Rover.is_stuck += 1
                if Rover.is_stuck > 100:
                    Rover.mode = 'stuck'
            else:
                Rover.is_stuck = 0
            # This helps to keep our rover from circling endlessly by switching on turning mode if the rover has
            # maintained a steering angle of 10 degree or more for a period of time.
            if Rover.steer > 10:
                Rover.turning += 1
                if Rover.turning > 200:
                    Rover.mode = 'turning'
            else:
                Rover.turning = 0
            if len(Rover.nav_angles) >= Rover.stop_forward:
                # If mode is forward, navigable terrain looks good
                # and velocity is below max, then throttle
                if Rover.vel < Rover.max_vel:
                    # Set throttle value to throttle setting
                    Rover.throttle = Rover.throttle_set
                else: # Else coast
                    Rover.throttle = 0
                Rover.brake = 0

                # This is my attempt at left wall hugging. Instead of grabbing the mean angle, I sort the angles, and then
                # grab the second half of the angles. This gives us the angles closest to the left wall.
                Rover.nav_angles = np.sort(Rover.nav_angles)[-int(len(Rover.nav_angles) * 0.5):]

                # This was a trial and error of trying to keep the rover from swaying back and forth dramatically when
                # space opened up on the right side when traveling down the left wall. In this first conditional, if we
                # are traveling down the left wall and we near a dead in, the max angle will tend closer to 0. In that case,
                # we should allow the rover to have a steer angle of -15 degrees (We don't want this during straight travel,
                # because it will start to stray).
                if np.max(Rover.nav_angles * 180/np.pi) < 25:
                    mean_angle = np.clip(np.mean(Rover.nav_angles * 180 / np.pi), -15, 15)
                # If we are on a straight away, we don't want to consider angles less than 0, as that will induce swaying.
                # This may not be the best solution, but I found it to be effective in simulation.
                else:
                    mean_angle = np.clip(np.mean(Rover.nav_angles * 180/np.pi), 0, 15)
                # Set steer angle
                Rover.steer = mean_angle
            # If there's a lack of navigable terrain pixels then go to 'stop' mode
            elif len(Rover.nav_angles) < Rover.stop_forward:
                    # Set mode to "stop" and hit the brakes!
                    Rover.throttle = 0
                    # Set brake to stored brake value
                    Rover.brake = Rover.brake_set
                    Rover.steer = 0
                    Rover.mode = 'stop'
            elif Rover.near_sample:
                Rover.throttle = 0
                Rover.brake = Rover.brake_set
                Rover.steer = 0
                Rover.mode = 'stop'

        # If we go over the turning threshold, we need to correct. Based on the speed of the rover, we reverse the turn
        # for a certain amount of time. The faster the rover, the less time we spend correcting.
        elif Rover.mode == 'turning':
            Rover.correcting_loop += 1
            Rover.brake = 0
            Rover.throttle = 0
            Rover.steer = -15
            if Rover.vel > 0.5:
                if Rover.correcting_loop > 50:
                    Rover.mode = 'forward'
                    Rover.correcting_loop = 0
            else:
                if Rover.correcting_loop > 100:
                    Rover.mode = 'forward'
                    Rover.correcting_loop = 0

        # If we go over the stuck threshold, we need to try to try turning into another angle until we can move forward
        # again. Again, there is probably a more clever way to accomplish this, but this was effective in simulation.
        elif Rover.mode == 'stuck':
            Rover.throttle = 0
            Rover.steer = -15
            Rover.is_stuck += 1
            if Rover.is_stuck > 125:
                Rover.is_stuck = 0
                Rover.mode = 'forward'


        # If we're already in "stop" mode then make different decisions
        elif Rover.mode == 'stop':
            # If we're in stop mode but still moving keep braking
            if Rover.vel > 0.2:
                Rover.throttle = 0
                Rover.brake = Rover.brake_set
                Rover.steer = 0
            # If we're not moving (vel < 0.2) then do something else
            elif Rover.vel <= 0.2:
                # Now we're stopped and we have vision data to see if there's a path forward
                if len(Rover.nav_angles) < Rover.go_forward:
                    Rover.throttle = 0
                    # Release the brake to allow turning
                    Rover.brake = 0
                    # Turn range is +/- 15 degrees, when stopped the next line will induce 4-wheel turning
                    Rover.steer = -15 # Could be more clever here about which way to turn
                # If we're stopped but see sufficient navigable terrain in front then go!
                if len(Rover.nav_angles) >= Rover.go_forward:
                    # Set throttle back to stored value
                    Rover.throttle = Rover.throttle_set
                    # Release the brake
                    Rover.brake = 0
                    # Set steer to mean angle
                    Rover.steer = np.clip(np.mean(Rover.nav_angles * 180/np.pi), -15, 15)
                    Rover.mode = 'forward'
    # Just to make the rover do something
    # even if no modifications have been made to the code
    else:
        Rover.throttle = Rover.throttle_set
        Rover.steer = 0
        Rover.brake = 0

    return Rover

