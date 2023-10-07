import numpy as np
import math

def action_generator(current_position, goal):  # under local frame [y, x, o] and o is degree, unit is m
    turning_threshold = 15
    current_location = np.array(current_position[:2])
    current_orientation = current_position[-1]
    while current_orientation >= 360 or current_orientation < 0:
        if current_orientation > 0:
            current_orientation -= 360
        else:
            current_orientation += 360
    #cos_current_orientation = np.cos(current_orientation)
    goal = np.array(goal)
    diff = goal - current_location
    #goal_tan_y = diff[1] / diff[0]
    goal_angle = arc_cos(diff[1], diff[0])
    #print(diff)
    #print(goal_angle)
    #print(current_orientation)
    #goal_angle = np.rad2deg(np.arctan(goal_tan_y))
    angle_diff = current_orientation - goal_angle

    # next we will confine the angle diff to [-180~+180)
    if angle_diff >= 180:
        angle_diff = angle_diff - 360
    if angle_diff < -180:
        angle_diff = 360 + angle_diff
    #print(angle_diff)
    """print('--')
    print('goal:{}'.format(goal))
    print('current_location:{}'.format(current_location))
    print('diff:{}'.format(diff))
    print('goal_angle:{}'.format(goal_angle))
    print('current_ori:{}'.format(current_orientation))
    print('anglediff:{}'.format(angle_diff))
    print('--')"""
    if angle_diff < turning_threshold and angle_diff > -turning_threshold:
        #print('move forward')
        return 2  # move forward
    if angle_diff <= -turning_threshold:
        #print('turn left')
        return 0  # turn left
    if angle_diff >= turning_threshold:
        #print('turn right')
        return 1  # turn right


def arc_cos(x, y):
    cos = y/math.sqrt(x**2+y**2)
    angle = np.rad2deg(np.arccos(cos))
    if x >= 0:
        return angle
    if x < 0:
        return 360-angle

if __name__ == '__main__':
    current_position = [284, 357, -300]
    goal = [275, 353]
    print(action_generator(current_position, goal))
    #print(arc_cos(0,-1))