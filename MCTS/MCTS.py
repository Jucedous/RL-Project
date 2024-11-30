import time
import math
import random
import numpy
from functools import partial
import copy
from MCTS.tree import treeNode


def get_next_steps_roll(y: str, step_n: int, mcts_task):
    # Generate the next steps for the roll-out phase
    next_steps = []
    for i in range(mcts_task.roll_branch):
        proposal = ''
        cnt = 3
        while not proposal and cnt:
            # Get the next step
            proposal = mcts_task.get_next_step(y, step_n)
            cnt -= 1
        if not proposal:
            continue
        next_steps.append(proposal)
    return next_steps


def get_next_steps_expand(node: treeNode, mcts_task):
    # Generate the next steps for the expansion phase
    next_steps = []
    reflection = node.reflection
    for i in range(mcts_task.branch):
        proposal = ''
        cnt = 3
        while not proposal and cnt:
            if mcts_task.use_reflection == 'common':
                # Get the next step using reflection
                proposal = mcts_task.get_next_step_use_reflection(node.y, node.depth + 1, reflection)
            else:
                # Get the next step without using reflection
                proposal = mcts_task.get_next_step(node.y, node.depth + 1)
            cnt -= 1
        if not proposal:
            continue
        next_steps.append(proposal)
    return next_steps


def randomPolicy(node: treeNode, mcts_task):
    # Apply a random policy to simulate the search
    max_V = mcts_task.low
    strs = node.y
    cur_step = node.depth + 1
    if mcts_task.use_reflection == 'common':
        # Get reflection using the common method
        reflection = mcts_task.get_reflection(strs, cur_step)
    else:
        # Get reflection using the simple method
        reflection = mcts_task.get_simple_reflection(strs, cur_step)
    node.update_reflection(reflection)
    if reflection == '<end>':
        # If the step is resolved, return the node's value
        print('This step has been resolved and does not require simulation.\n')
        return node.V
    for i in range(mcts_task.roll_forward_steps):
        # Get the next steps for the roll-out
        next_steps = get_next_steps_roll(strs, cur_step, mcts_task)
        if not next_steps:
            break
        # Choose a random action from the next steps
        action = random.choice(next_steps)
        strs = strs + action
        cur_step += 1
        # Get the value of the current step
        value = mcts_task.get_step_value(strs)
        if value > max_V:
            max_V = value
        if mcts_task.use_reflection == 'common':
            # Get the current reflection using the common method
            cur_ref = mcts_task.get_reflection(strs, cur_step)
        else:
            # Get the current reflection using the simple method
            cur_ref = mcts_task.get_simple_reflection(strs, cur_step)
        if cur_ref == '<end>':
            break
    return max_V


def greedyPolicy(node: treeNode, mcts_task):
    # Apply a greedy policy to simulate the search
    max_V = mcts_task.low
    strs = node.y
    cur_step = node.depth + 1
    if mcts_task.use_reflection == 'common':
        # Get reflection using the common method
        reflection = mcts_task.get_reflection(strs, cur_step)
    else:
        # Get reflection using the simple method
        reflection = mcts_task.get_simple_reflection(strs, cur_step)
    node.update_reflection(reflection)
    if reflection == '<end>':
        # If the step is resolved, return the node's value
        print('This step has been resolved and does not require simulation.\n')
        return node.V
    for i in range(mcts_task.roll_forward_steps):
        # Get the next steps for the roll-out
        actions = get_next_steps_roll(strs, cur_step, mcts_task)
        if not actions:
            break
        # Generate new strings by appending each action
        new_ys = [strs + action for action in actions]
        cur_step += 1
        # Get the values of the new strings
        values = [mcts_task.get_step_value(new_y) for new_y in new_ys]
        # Choose the action with the maximum value
        idx = numpy.argmax(values)
        strs = new_ys[idx]
        value = values[idx]
        if value > max_V:
            max_V = value
        if mcts_task.use_reflection == 'common':
            # Get the current reflection using the common method
            cur_ref = mcts_task.get_reflection(strs, cur_step)
        else:
            # Get the current reflection using the simple method
            cur_ref = mcts_task.get_simple_reflection(strs, cur_step)
        if cur_ref == '<end>':
            break
    return max_V


def MCTS_search(mcts_task):
    # Initialize the root node
    root = treeNode('')

    if mcts_task.limit_type == 'time':
        # Set the time limit for the search
        timeLimit = time.time() + mcts_task.time_limit / 1000
        time_start = time.time()
        while time.time() < timeLimit:
            # Print the start of a new search with the elapsed time
            print(f'<Start New Seach, total time:{time.time() - time_start}>\n')
            # Execute a round of MCTS
            flag, node, root = executeRound(root, mcts_task)
            # If a solution is found, print a message and return the results
            if flag:
                print('Solution Found！\n')
                return root, node, time.time() - time_start
    else:
        # Iterate for a fixed number of rounds if time limit is not used
        for i in range(mcts_task.iteration_limit):
            # Print the start of a new search round with the current round number
            print(f'<Start New Search Round, current round:{i}>\n')
            # Execute a round of MCTS
            flag, node, root = executeRound(root, mcts_task)
            # If a solution is found, print a message and return the results
            if flag:
                print('Solution Found！\n')
                return root, node, i + 1
    # Return the root, None for node, and None for time if no solution is found
    return root, None, None


def executeRound(root, mcts_task):
    # Execute a selection-expansion-simulation-backpropagation round

    # Print a separator and the start of node selection
    print('-' * 40)
    print('Chossing Node\n')
    # Select a node
    flag, node = selectNode(root, mcts_task)
    # If a terminal node is found, handle it based on the sample value
    if flag:
        if mcts_task.sample_value != 'full':
            return True, node, root
        else:
            node.reflection = '<end>'

    # Print a separator and the start of node expansion
    print('-' * 40)
    print('Expanding\n')
    # Skip expansion if the node is terminal
    if node.reflection == '<end>':
        print('Skip\n')
    else:
        # Expand the node
        node = expand(node, mcts_task)

    # If the reward model type is 'vm', simulate the search
    if mcts_task.reward_model_type == 'vm':
        # Print a separator and the start of simulation
        print('-' * 40)
        print('Simulate Seach\n')
        # Skip simulation if the node is terminal
        if node.reflection == '<end>':
            print('Skip\n')
        else:
            # Get the best child node
            roll_node = getBestChild(node, mcts_task)
            # Apply the appropriate policy to get the best value
            best_V = greedyPolicy(roll_node, mcts_task) if mcts_task.roll_policy == 'greedy' else randomPolicy(roll_node, mcts_task)
            # Update the value and visit count of the roll node
            roll_node.V = roll_node.V * (1 - mcts_task.alpha) + best_V * mcts_task.alpha
            roll_node.numVisits += 1

    # Print a separator and the start of backpropagation
    print('-' * 40)
    print('back_propagate\n')
    # Backpropagate the results
    back_propagate(node)
    # Return the flag, node, and root
    return False, node, root


def isTerminal(node, mcts_task):
    # Check if the node is terminal based on the reward model type
    if mcts_task.reward_model_type == 'vm':
        return node.V >= mcts_task.end_gate
    else:
        return False


def selectNode(node, mcts_task):
    # Select a node for expansion
    while node.isFullyExpanded:
        node = getBestChild(node, mcts_task)
    if isTerminal(node, mcts_task):
        node.final_ans_flag = 1
        return True, node
    else:
        return False, node


def expand(node: treeNode, mcts_task):
    # Expand the selected node
    if not node.reflection:
        if mcts_task.use_reflection == 'common':
            reflection = mcts_task.get_reflection(node.y, node.depth + 1)
        else:  # simple
            reflection = mcts_task.get_simple_reflection(node.y, node.depth + 1)
        node.update_reflection(reflection)
    if node.reflection == '<end>':
        return node
    actions = get_next_steps_expand(node, mcts_task)
    if not actions:
        node.update_reflection('<end>')
        return node

    for action in actions:
        if action not in node.children.keys():
            node.append_children(action)
            child = node.children[action]
            value = mcts_task.get_step_value(child.y)
            child.update_value(value)
            if mcts_task.sample_value == 'full':
                if mcts_task.use_reflection == 'common':
                    child.update_reflection(mcts_task.get_reflection(child.y, child.depth + 1))
                else:
                    child.update_reflection(mcts_task.get_simple_reflection(child.y, child.depth + 1))
            child.visit_sequence = mcts_task.node_count
            mcts_task.update_count()
    node.isFullyExpanded = True
    return node


def back_propagate(node):
    # Backpropagate the results of the simulation
    while node is not None:
        node.numVisits += 1
        if node.isFullyExpanded:
            child_Vs = [child.V * child.numVisits for child in node.children.values()]
            total_num_visits = sum([child.numVisits for child in node.children.values()])
            if total_num_visits > 0:
                node.V = sum(child_Vs) / total_num_visits
        node = node.parent


def getBestChild(node, mcts_task):
    # Get the best child node based on the UCT value
    bestValue = mcts_task.low
    bestNodes = []
    for child in node.children.values():
        nodeValue = child.V + mcts_task.exploration_constant * math.sqrt(
            2 * math.log(node.numVisits) / child.numVisits) if child.numVisits > 0 else child.V + mcts_task.INF
        if nodeValue > bestValue:
            bestValue = nodeValue
            bestNodes = [child]
        elif nodeValue == bestValue:
            bestNodes.append(child)
    return random.choice(bestNodes)


def MCTS(mcts_task):
    # Perform the MCTS search
    root, node, finish = MCTS_search(mcts_task)

    if mcts_task.sample_value == 'full':
        print('Sampling Finished\n')
        return None, -1, root
    else:
        if mcts_task.reward_model_type == 'vm':
            if finish is not None:
                print(f'Solution Found!\nSolution:{node.y}\n')
                return node, finish, root

            else:
                best_node, best_V = root.getBestV()
                print(f'Highest Value Instead\nSolution:{best_node.y}\n')
                return best_node, -1, root
        else:
            print('Not Supported\n')
            return None, -1, root