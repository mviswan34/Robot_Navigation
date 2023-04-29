import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import io
import time
from warehouse_robot import get_shortest_path, get_starting_location, is_terminal_state, get_next_location, get_next_action, q_values, rewards  # assuming your Q-Learning code is in warehouse_robot.py
import pandas as pd

def plot_environment(robot_row, robot_col):
    plt.figure(dpi=100)  
    warehouse_env = np.zeros((11, 11, 3))
    warehouse_env.fill(255)
    warehouse_env[rewards == -100] = [255, 0, 0]
    warehouse_env[0, 6] = [0, 255, 0]

    plt.imshow(warehouse_env)
    plt.xticks(range(11))
    plt.yticks(range(11))
    
    # Create a blue circle patch for the robot
    robot_circle = patches.Circle((robot_col, robot_row), radius=0.3, facecolor='blue')
    plt.gca().add_patch(robot_circle)

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()

    return buf.getvalue()

def visualize_q_table(q_values, action_symbols=['↑', '→', '↓', '←']):
    rows, cols, _ = q_values.shape
    q_table_str = []

    for r in range(rows):
        row = []
        for c in range(cols):
            max_q_value_index = np.argmax(q_values[r, c])
            max_q_value = q_values[r, c, max_q_value_index]
            row.append(f"{action_symbols[max_q_value_index]} {max_q_value:.2f}")
        q_table_str.append(row)

    q_table_df = pd.DataFrame(q_table_str, columns=range(cols), index=range(rows))
    return q_table_df


def finish_training(val):
    #define training parameters
    epsilon = 0.9 #the percentage of time when we should take the best action (instead of a random action)
    discount_factor = 0.9 #discount factor for future rewards
    learning_rate = 0.9 #the rate at which the AI agent should learn
    warehouse_placeholder = st.empty()
    q_table_placeholder = st.empty()
    for episode in range(1000):
        #get the starting location for this episode
        row_index, column_index = get_starting_location()
        #continue taking actions (i.e., moving) until we reach a terminal state
        #(i.e., until we reach the destination or crash into an obstacle)
        while not is_terminal_state(row_index, column_index):
            #choose which action to take (i.e., where to move next)
            action_index = get_next_action(row_index, column_index, epsilon)
            #perform the chosen action, and transition to the next state (i.e., move to the next location)
            old_row_index, old_column_index = row_index, column_index #store the old row and column indexes
            row_index, column_index = get_next_location(row_index, column_index, action_index)    
            #receive the reward for moving to the new state, and calculate the temporal difference
            reward = rewards[row_index, column_index]
            old_q_value = q_values[old_row_index, old_column_index, action_index]
            temporal_difference = reward + (discount_factor * np.max(q_values[row_index, column_index])) - old_q_value
            #update the Q-value for the previous state and action pair
            new_q_value = old_q_value + (learning_rate * temporal_difference)
            q_values[old_row_index, old_column_index, action_index] = new_q_value
            if val == 1 :
                if episode in [900,999]:
                    warehouse_env = plot_environment(row_index, column_index)
                    q_table_df = visualize_q_table(q_values)
                    # Display warehouse environment and Q-Table
                    warehouse_placeholder.image(warehouse_env, caption=f"Episode: {episode}", width=400)
                    q_table_placeholder.dataframe(q_table_df)
                    time.sleep(0.2)

    st.success("Training Completed !!!")


st.title("Navigating the Robot through an Obstacle Course Using Q-Learning")
tab1, tab2 = st.tabs(["Application", "About Q-Learning"])
st.subheader("Train you model: ")
if st.button("Visualize and Train"):
        param = 1
        finish_training(param)
elif st.button("Directly Train"):
        param = 0
        finish_training(param)
        
st.markdown("Choose your starting position")
start_row = st.slider("Row", 0, 10, 1)
start_col = st.slider("Column", 0, 10, 1)

if st.button("Find shortest path"):
        # Get the shortest path from the starting location
        shortest_path = get_shortest_path(start_row, start_col)

        progress_bar = st.progress(0)
        num_steps = len(shortest_path)

        # Create a placeholder for the visualization
        image_placeholder = st.empty()

        # Visualize the shortest path
        for idx, position in enumerate(shortest_path):
            warehouse_env = plot_environment(position[0], position[1])
            progress_bar.progress((idx + 1) / num_steps)

            # Update the image in the placeholder
            image_placeholder.image(warehouse_env, width=400)

