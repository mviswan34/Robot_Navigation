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
    #The exploration rate
    epsilon = 0.9 
    #Discount factor for future rewards
    discount_factor = 0.9 
    #Learning rate of the agent
    learning_rate = 0.9
    warehouse_placeholder = st.empty()
    st.markdown("Watch closely as the robot navigates 🤖🛣️, updating its Q-values below! With an exploration rate of 0.9 and an exploitation rate of 0.1, our curious robot loves discovering new paths 🗺️. The 0.9 discount rate means it values future rewards almost as much as the ones right in front of it 🎁. And with a learning rate of 0.9, our savvy agent wastes no time mastering its environment! 🎓💨")
    q_table_placeholder = st.empty()
    for episode in range(1000):
        #Get the location to begin exploring from
        row_index, column_index = get_starting_location()
        #Until the destination is reached or the robot crashes into an obstacle keep moving
        while not is_terminal_state(row_index, column_index):
            #Choose the action to take
            action_index = get_next_action(row_index, column_index, epsilon)
            #Move to the next state based on chosen action
            #Store the old row and column indices , so as to update the previous state using temporal difference
            old_row_index, old_column_index = row_index, column_index 
            row_index, column_index = get_next_location(row_index, column_index, action_index)    
            #Get the reward
            reward = rewards[row_index, column_index]
            #Calculate Temporal Difference
            old_q_value = q_values[old_row_index, old_column_index, action_index]
            temporal_difference = reward + (discount_factor * np.max(q_values[row_index, column_index])) - old_q_value
            #Update the Q-value of the previous State-Action Pair in the Q-Table
            new_q_value = old_q_value + (learning_rate * temporal_difference)
            q_values[old_row_index, old_column_index, action_index] = new_q_value
            #Check if the user opted for Visualization while training
            if val == 1 :
                #If so visualize only for few episodes
                if episode in [1,10,20,30,40,100,200,300,400,500,600,700,800,900,999]:
                    warehouse_env = plot_environment(row_index, column_index)
                    q_table_df = visualize_q_table(q_values)
                    # Display warehouse environment and Q-Table
                    warehouse_placeholder.image(warehouse_env, caption=f"Episode: {episode}", width=400)
                    q_table_placeholder.dataframe(q_table_df)
                    time.sleep(0.2)

    st.success("🌟🤖 Training Complete! Your AI agent is now ready for action! 🤖🌟")


st.title("Mastering Obstacle Course Navigation: A Q-Learning Approach for Autonomous Robots")
tab1, tab2 = st.tabs(["Application", "About Q-Learning"])
with tab1:
    st.markdown("Train you model: ")
    col3,col4 = st.columns(2)
    if col3.button("Visualize and Train"):
        st.write("Note that due to the time-consuming nature of visualizing every episode, we have chosen to display the visualizations for episodes 1, 10, 20, 30, 40, and every 100th episode thereafter.")
        param = 1
        finish_training(param)
        st.markdown("🚀 Fasten your seatbelts and get ready for a thrilling experience as the AI agent showcases its newly-acquired skills. Watch in awe as the agent expertly avoids obstacles, makes strategic decisions, and reaches its destination with remarkable efficiency.")
    elif col4.button("Skip Visualization"):
        st.markdown("🏃‍♂️💨 Skipping Visualization: AI Agent in Stealth Training Mode 🕶️🤖")
        st.markdown("Your AI agent is currently undergoing an intense training regimen for 1000 episodes, sharpening its skills and honing its strategies to navigate the warehouse environment with the utmost precision.")
        st.markdown("🔜 Once the training is complete, your AI agent will emerge as a formidable force, equipped with the knowledge and experience to tackle any task that comes its way. Stay tuned and prepare to be amazed!")
        param = 0
        finish_training(param)
        
    st.markdown("Select Your Launchpad: Pick the AI Agent's Starting Point!")
    st.markdown("Remember to pick a clear spot for your starting point, or else there won't be anything to show! 🚀")
    st.markdown("Need some guidance? 🧭 Check out the handy reference plot on the right-hand side! 👉")
    warehouse_image = plot_environment(1, 0)
    col6,col7 = st.columns(2)
    start_row = col6.slider("Row", 0, 10, 1)
    start_col = col6.slider("Column", 0, 10, 1)
    # Display the environment using Streamlit's st.image
    col7.image(warehouse_image, caption="Warehouse Environment", width=400)

    if col6.button("Find shortest path"):
        
        # Get the shortest path from the starting location
        shortest_path = get_shortest_path(start_row, start_col)
        if len(shortest_path) == 0:
                st.markdown("Oops! 🚧 You've picked an obstacle as your starting point. Please give it another shot and choose a valid position! 🎯")

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
with tab2:
    st.header("What is Q-Learning?")
    col1,col2 = st.columns(2)
    image1 = "https://hub.packtpub.com/wp-content/uploads/2019/12/reinforcement-learning-1024x835.png"
    col1.image(image1)
    col2.markdown("Q-Learning is a type of machine learning algorithm that enables an agent to learn through trial and error by interacting with its environment. It is a model-free approach, meaning that it does not require a pre-existing model of the environment or task to be learned. Instead, Q-Learning allows the agent to learn by estimating the expected rewards for each action it can take in a given state. This approach is particularly useful for complex tasks where it may not be feasible to define an explicit model of the environment.")
