import numpy as np
import streamlit as st
#Trials with multiple mazes
def maze1():
    aisles = {i: [] for i in range(11)}
    aisles[0] = [0,1,2,4,6,7,10]
    aisles[1] = [0,2,3,4,5,6,7,8,9,10]
    aisles[2] = [0,1,2,6,8,9]
    aisles[3] = [0,2,3,5,6,7,9,10]
    aisles[4] = [0,1,3,4,5,7,9,10]
    aisles[5] = [1,2,3,7,8,10]
    aisles[6] = [0,1,3,4,5,6,8,9,10]
    aisles[7] = [0,3,4,5,6,7,8,10]
    aisles[8] = [0,1,3,4,6,7,8,10]
    aisles[9] = [1,2,3,4,6,7,8,9,10]
    aisles[10] = [2,3,5,6,9,10]
    return aisles

def maze2():
    aisles = {i: [] for i in range(11)}
    aisles[0] = [2,3,4,5,6,8,9,10]
    aisles[1] = [0,1,2,4,6,7,8,10]
    aisles[2] = [0,2,3,4,8,10]
    aisles[3] = [0,1,4,5,6,7,8,10]
    aisles[4] = [1,2,3,6,7,10]
    aisles[5] = [0,3,4,5,6,7,8,9,10]
    aisles[6] = [0,1,2,3,6,9]
    aisles[7] = [3,5,6,7,8,9,10]
    aisles[8] = [0,1,2,3,5,8,9]
    aisles[9] = [0,3,4,5,6,7,8,9,10]
    aisles[10] = [0,1,2,3,6,7,9,10]
    return aisles
def maze3():
    aisles = {}
    aisles[0] = [0,1,2,4,6,7,10]
    aisles[1] = [0,1,2,3,5,6,7,8,9,10]
    aisles[2] = [3,4,5,6,7,10]
    aisles[3] = [0,1,2,3,4,7,8,9,10]
    aisles[4] = [0,4,5,6,7,8,9,10]
    aisles[5] = [0,1,2,3,4,6,]
    aisles[6] = [3,4,6,7,8,9,10]
    aisles[7] = [0,1,2,3,4,5,6,8,9]
    aisles[8] = [1,3,6,7,9,10]
    aisles[9] = [0,1,2,3,4,5,6,7,8,9,10]
    aisles[10] = [2,3,5,6,9,10]
    return aisles


al = maze1()

#Create a 2D numpy array to hold the rewards for each state. 
def initialize_rewards(environment_rows, environment_columns,aisles):
    rw = np.full((environment_rows, environment_columns), -100.)
    rw[0, 6] = 100.  # Set the reward for the packaging area (i.e., the goal) to 100
    for row_index in range(1, 10):
        for column_index in aisles[row_index]:
            rw[row_index, column_index] = -1.
    return rw

environment_rows = 11
environment_columns = 11
#Set the initial Q-Values
q_values = np.zeros((environment_rows, environment_columns, 4))
#numeric action codes: 0 = up, 1 = right, 2 = down, 3 = left
actions = ['up', 'right', 'down', 'left']
#Get the reward matrix based on the environment defined.
rewards = initialize_rewards(environment_rows, environment_columns,al)


#Check if the state is a terminal state or not
def is_terminal_state(current_row_index, current_column_index):
  if rewards[current_row_index, current_column_index] == -1.:
    return False
  else:
    return True

#Select a random starting location which is not terminal
def get_starting_location():
  current_row_index = np.random.randint(environment_rows)
  current_column_index = np.random.randint(environment_columns)
  while is_terminal_state(current_row_index, current_column_index):
    current_row_index = np.random.randint(environment_rows)
    current_column_index = np.random.randint(environment_columns)
  return current_row_index, current_column_index

#The epsilon greedy algorithm to choose which action to take next
def get_next_action(current_row_index, current_column_index, epsilon):
  if np.random.random() < epsilon:
    return np.argmax(q_values[current_row_index, current_column_index])
  else: 
    return np.random.randint(4)

#Get the next location based on the chosen action
def get_next_location(current_row_index, current_column_index, action_index):
  new_row_index = current_row_index
  new_column_index = current_column_index
  if actions[action_index] == 'up' and current_row_index > 0:
    new_row_index -= 1
  elif actions[action_index] == 'right' and current_column_index < environment_columns - 1:
    new_column_index += 1
  elif actions[action_index] == 'down' and current_row_index < environment_rows - 1:
    new_row_index += 1
  elif actions[action_index] == 'left' and current_column_index > 0:
    new_column_index -= 1
  return new_row_index, new_column_index

#Define a function that will get the shortest path between the specified location and the destination.
def get_shortest_path(start_row_index, start_column_index):
  #Return empty if the location is that of a terminal state
  if is_terminal_state(start_row_index, start_column_index):
    return []
  else:
    current_row_index, current_column_index = start_row_index, start_column_index
    shortest_path = []
    shortest_path.append([current_row_index, current_column_index])
    #While until you reach the destination
    while not is_terminal_state(current_row_index, current_column_index):
      #Take the best action
      action_index = get_next_action(current_row_index, current_column_index, 1.)
      #Move to the next state and add the new location to the shortest_path list
      current_row_index, current_column_index = get_next_location(current_row_index, current_column_index, action_index)
      shortest_path.append([current_row_index, current_column_index])
    return shortest_path
