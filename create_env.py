import csv
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

path1 = 'position_record/drone_positions_best_env1.csv'
path2 = 'position_record/drone_positions_sddpg_env4v3.csv'
path3 = 'position_record/drone_positions_ddpg_env2.csv'
path4 = 'position_record/moving_target.csv'
path5 = 'position_record/drone_positions_best_env1_expand.csv'

method1_df = pd.read_csv(path1)
method2_df = pd.read_csv(path2)
method3_df = pd.read_csv(path3)
method4_df = pd.read_csv(path4)
method5_df = pd.read_csv(path5)

def plot_environment1():
    fig, ax = plt.subplots()

    # Define the positions and sizes of the obstacles
    obstacles = [
        (-42, 25, 9, 9),
        (-17, 25, 9, 9),
        (8, 25, 9, 9),
        (33, 25, 9, 9),
        (-29.5, 50, 9, 9),
        (-4.5, 50, 9, 9),
        (20.5, 50, 9, 9),
        (-17, 75, 9, 9),
        (8, 75, 9, 9),
    ]
    
    # Draw the obstacles
    for (x, y, w, h) in obstacles:
        ax.add_patch(patches.Rectangle((x, y), w, h, edgecolor='gray', facecolor='gray'))

    # Draw the starting point
    ax.add_patch(patches.Rectangle((-10, 0), 20, 10, edgecolor='green', facecolor='green'))

    # Draw the target point
    ax.add_patch(patches.Rectangle((-20, 90), 40, 10, edgecolor='red', facecolor='red'))

    # Draw the drone path
    # ax.plot(method3_df['y'], method3_df['x'], label=f'DDPG: {len(method3_df)} steps', marker='o', color='purple', markersize=3, linestyle='-', linewidth=2)
    # ax.plot(method2_df['y'], method2_df['x'], label=f'Base: {len(method2_df)} steps', marker='o', color='orange', markersize=3, linestyle='-', linewidth=2)
    ax.plot(method5_df['y'], method5_df['x'], label=f'Ours EX: {len(method5_df)} steps', marker='o', color='orange', markersize=3, linestyle='-', linewidth=2)
    ax.plot(method1_df['y'], method1_df['x'], label=f'Ours: {len(method1_df)} steps', marker='o', color='blue', markersize=3, linestyle='-', linewidth=2)
    ax.legend(fontsize=8)

    ax.set_xlim(-50, 50)
    ax.set_ylim(0, 100)
    ax.set_aspect('equal')
    ax.grid(True)

    plt.xlabel('X axis (meters)')
    plt.ylabel('Y axis (meters)')
    plt.title('2D Visualization of the Expand Environment 1')
    # plt.legend()
    plt.show()


def plot_environment2():

    fig, ax = plt.subplots()

    # Define the positions and sizes of the obstacles
    obstacles = [
        (-42, 25, 9, 9), #1
        (-17, 25, 9, 9), #2
        (8, 25, 9, 9),   #3
        (33, 25, 9, 9),  #4
        (-29.5, 50, 9, 9), #5
        (20.5, 50, 9, 9),  #7
        (-17, 75, 9, 9),   #9
        (-8, 75, 16, 9),   #10 (special case
        (8, 75, 9, 9),     #11
    ]

    circle_obstacles = [
        (0, 54.5, 4.5), #6
        (-37.5, 79.5, 4.5), #8
        (37.5, 79.5, 4.5), #12
    ]

    # Draw the obstacles
    for idx, (x, y, w, h) in enumerate(obstacles):
        if idx == 7:
            ax.add_patch(patches.Rectangle((x, y), w, h, edgecolor='gray', facecolor='none'))
        else:
            ax.add_patch(patches.Rectangle((x, y), w, h, edgecolor='gray', facecolor='gray'))

    for (x, y, r) in circle_obstacles:
        ax.add_patch(patches.Circle((x, y), r, edgecolor='gray', facecolor='gray'))

    # Draw the starting point
    ax.add_patch(patches.Rectangle((-10, 0), 20, 10, edgecolor='green', facecolor='green'))

    # Draw the target point
    ax.add_patch(patches.Rectangle((-20, 90), 40, 10, edgecolor='red', facecolor='red'))

    # Draw the drone path
    ax.plot(method3_df['y'], method3_df['x'], label=f'DDPG: {len(method3_df)} steps', marker='o', color='purple', markersize=3, linestyle='-', linewidth=2)
    ax.plot(method2_df['y'], method2_df['x'], label=f'Base: {len(method2_df)} steps', marker='o', color='orange', markersize=3, linestyle='-', linewidth=2)
    ax.plot(method1_df['y'], method1_df['x'], label=f'Ours: {len(method1_df)} steps', marker='o', color='blue', markersize=3, linestyle='-', linewidth=2)
    ax.legend(fontsize=9)

    ax.set_xlim(-50, 50)
    ax.set_ylim(0, 100)
    ax.set_aspect('equal')
    ax.grid(True)

    plt.xlabel('X axis (meters)')
    plt.ylabel('Y axis (meters)')
    plt.title('2D Visualization of the Environment 2')
    # plt.legend()
    plt.show()


def plot_environment3():

    fig, ax = plt.subplots()

    # Define the positions and sizes of the obstacles
    obstacles = [
        (-29.5, 25, 9, 9),
        (-4.5, 25, 9, 9),
        (20.5, 25, 9, 9),
        (-42, 50, 9, 9),
        (-17, 50, 9, 9),
        (-8, 50, 16, 9),
        (8, 50, 9, 9),
        (33, 50, 9, 9),
        (-29.5, 75, 9, 9),
        (-4.5, 75, 9, 9),
        (20.5, 75, 9, 9),
    ]

    # Draw the obstacles
    for idx, (x, y, w, h) in enumerate(obstacles):
        if idx == 5:
            ax.add_patch(patches.Rectangle((x, y), w, h, edgecolor='gray', facecolor='none'))
        else:
            ax.add_patch(patches.Rectangle((x, y), w, h, edgecolor='gray', facecolor='gray'))

    # Draw the starting point
    ax.add_patch(patches.Rectangle((-10, 0), 20, 10, edgecolor='green', facecolor='green'))

    # Draw the target point
    ax.add_patch(patches.Rectangle((-20, 90), 40, 10, edgecolor='red', facecolor='red'))

    # Draw the drone path
    ax.plot(method3_df['y'], method3_df['x'], label=f'DDPG: {len(method3_df)} steps', marker='o', color='purple', markersize=3, linestyle='-', linewidth=2)
    ax.plot(method2_df['y'], method2_df['x'], label=f'Base: {len(method2_df)} steps', marker='o', color='orange', markersize=3, linestyle='-', linewidth=2)
    ax.plot(method1_df['y'], method1_df['x'], label=f'Ours: {len(method1_df)} steps', marker='o', color='blue', markersize=3, linestyle='-', linewidth=2)
    ax.legend(fontsize=9)

    ax.set_xlim(-50, 50)
    ax.set_ylim(0, 100)
    ax.set_aspect('equal')
    ax.grid(True)

    plt.xlabel('X axis (meters)')
    plt.ylabel('Y axis (meters)')
    plt.title('2D Visualization of the Environment 3')
    # plt.legend()
    plt.show()

def plot_mtenvironment():
    fig, ax = plt.subplots()

    # Define the positions and sizes of the obstacles
    obstacles = [
        (-42, 25, 9, 9),
        (-17, 25, 9, 9),
        (8, 25, 9, 9),
        (33, 25, 9, 9),
        (-29.5, 50, 9, 9),
        (-4.5, 50, 9, 9),
        (20.5, 50, 9, 9),
        (-17, 75, 9, 9),
        (8, 75, 9, 9),
        (-29.5, 100, 9, 9),
        (-4.5, 100, 9, 9),
        (20.5, 100, 9, 9),
        (-42, 125, 9, 9),
        (-17, 125, 9, 9),
        (8, 125, 9, 9),
        (33, 125, 9, 9),
    ]

    # circle_target = [
    #     (-14.75, 85, 1),
    #     (10.25, 65,1),
    #     (-10.25, 62.25,1),
    #     (0, 95, 1),
    #     (-14.75, 85, 1),
    #     (10.25, 65,1),
    # ]

    circle_target = [
        (12.5, 100, 1),
        (12.5, 112.5, 1),
        (0, 125, 1),
        (0, 137.5, 1),
        (0, 150, 1),
        (-12.5, 100, 1),
        (-12.5, 112.5, 1),
        (-25, 125, 1),
        (-25, 137.5, 1),
        (-12.5, 142.5, 1),
        (25, 125, 1),
        (25, 137.5, 1),
        (12.5, 142.5, 1),
        (0, 87.5, 1)
    ]

    # for (x, y, r) in circle_target:
    #     ax.add_patch(patches.Circle((x, y), r, edgecolor='orange', facecolor='orange'))
    
    # Draw the obstacles
    for (x, y, w, h) in obstacles:
        ax.add_patch(patches.Rectangle((x, y), w, h, edgecolor='gray', facecolor='gray'))
    

    # Draw the starting point
    ax.add_patch(patches.Rectangle((-10, 0), 20, 10, edgecolor='green', facecolor='green'))

    # Draw the target point
    ax.add_patch(patches.Rectangle((-20, 145), 40, 10, edgecolor='red', facecolor='none', linewidth=2))

    # Draw the drone path
    ax.plot(method2_df['y'], method2_df['x'], label=f'Base: {len(method2_df)} steps', marker='o', color='purple', markersize=2, linestyle='-', linewidth=2)
    ax.plot(method1_df['y'], method1_df['x'], label=f'Ours: {len(method1_df)} steps', marker='o', color='blue', markersize=2, linestyle='-', linewidth=2)
    ax.plot(method4_df['x'], method4_df['y'], label=f'Moving Target', marker='o', color='orange', markersize=3, linestyle='-', linewidth=2)
    ax.legend(fontsize=9)

    ax.set_xlim(-100, 100)
    ax.set_ylim(0, 200)
    ax.set_aspect('equal')
    ax.grid(True)

    plt.xlabel('X axis (meters)')
    plt.ylabel('Y axis (meters)')
    plt.title('2D Visualization of the Environment 4')
    # plt.legend()
    plt.show()


plot_environment1()

