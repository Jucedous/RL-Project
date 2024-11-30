import copy
import numpy as np

class treeNode(object):
    def __init__(self, pcd, parent=None, depth=0):
        self.pcd = pcd  # str: The current node's identifier or data.
        self.y = ''  # str: The accumulated path from the root to this node.
        self.parent = parent  # treeNode: The parent node.
        self.numVisits = 0  # int: The number of times this node has been visited.
        self.V = 0  # float: The value of this node.
        self.children = {}  # dict{str:treeNode}: The children of this node.
        self.depth = depth  # int: The depth of this node in the tree.
        self.isFullyExpanded = False  # bool: Whether this node is fully expanded.
        self.visit_sequence = 0  # int: The sequence of visits to this node.
        self.final_ans_flag = 0  # int: Flag indicating if this node is a final answer.
        self.reflection = ''  # str: Reflection or additional information about this node.
        self.isTerminal = False  # bool: Whether this node is terminal.
        self.on_final_route = False  # bool: Whether this node is on the final route.
        self.min_steps_to_correct = 1024  # int: Minimum steps to correct from this node.
        self.summary = ''  # str: Summary information about this node.
        self.he = 0  # int: Hard estimation value.
        self.se = 0  # int: Soft estimation value.

    def __str__(self):
        # Provides a string representation of the node, including the number of visits, value, and possible actions (children).
        s = ["numVisits: %d" % self.numVisits, f'V:{self.V}', "possibleActions: %s" % (self.children.keys())]
        return "%s: {%s}" % (self.__class__.__name__, ', '.join(s))

    def append_children(self, new_pcd: str):
        # Adds a new child node to the current node.
        node = treeNode(new_pcd, self, self.depth + 1)
        node.update_y_from_parent()
        self.children.update({new_pcd: node})
        return self

    def update_y_from_parent(self):
        # Updates the y attribute based on the parent's y attribute.
        if self.parent is None:
            self.y = self.pcd
        else:
            self.y = self.parent.y + self.pcd

    def update_value(self, value):
        # Updates the value (V) of the node.
        self.V = value

    def update_reflection(self, reflection):
        # Updates the reflection attribute of the node.
        self.reflection = reflection

    def getBestV(self):
        # Gets the subtree maximum value node.
        if not self.isFullyExpanded:
            return self, self.V
        max_V = self.V
        max_node = self
        for child in self.children.values():
            subNode, subValue = child.getBestV()
            if subValue >= max_V:
                max_V = subValue
                max_node = subNode
        return max_node, max_V

    def trace_route(self):
        # Trace route from terminal node to root, marking nodes on the final route.
        cur_node = self
        while cur_node is not None:
            cur_node.on_final_route = True
            cur_node = cur_node.parent

    def get_new_value_samples(self):
        # Get value samples from search tree (start from terminal node).
        if self.depth == 0:
            return []
        step_value = 1.0 / self.depth
        new_samples = []
        cur_node = self.parent
        while cur_node is not None:
            for child in cur_node.children.values():
                if child.on_final_route:
                    child_value = step_value * child.depth
                    new_item = {'steps': child.y, 'value': child_value}
                    new_samples.append(new_item)
                else:
                    child_value = max(step_value * (cur_node.depth - 1), 0)
                    new_item = {'steps': child.y, 'value': child_value}
                    new_samples.append(new_item)
            cur_node = cur_node.parent
        return new_samples

    def get_all_end_root_nodes_vm(self, end_gate):
        # Retrieves all end root nodes based on a value threshold (end_gate).
        end_nodes = []
        if self.isFullyExpanded:
            for child in self.children.values():
                end_nodes.extend(child.get_all_end_root_nodes_vm(end_gate))
            return end_nodes
        else:
            if self.V >= end_gate or self.reflection == '<end>':
                return [self]
            else:
                return []

    def get_all_end_root_nodes_prm(self):
        # Retrieves all end root nodes based on the reflection attribute.
        end_nodes = []
        if self.isFullyExpanded:
            for child in self.children.values():
                end_nodes.extend(child.get_all_end_root_nodes_prm())
            return end_nodes
        else:
            if self.reflection == '<end>':
                return [self]
            else:
                return []

    def get_all_value_samples_vm(self):
        # Retrieves all value samples from the subtree.
        full_value_samples = []
        if self.depth == 0:
            self.V = 0
        else:
            if self.he == 0:
                r = -1
            else:
                r = 1
            self.V = max(0, (1 - self.parent.V) * r / self.min_steps_to_correct + self.parent.V)
            full_value_samples.append({'steps': self.y, 'value': self.V})
        if self.isFullyExpanded:
            for child in self.children.values():
                if child.min_steps_to_correct < 1024:
                    sub_samples = child.get_all_value_samples_vm()
                    full_value_samples.extend(sub_samples)
        return full_value_samples

    def get_full_value_samples_vm(self, end_leaf_nodes):
        # Retrieves full value samples from the subtree, considering end leaf nodes.
        for leaf in end_leaf_nodes:
            if leaf.min_steps_to_correct > 1:
                continue
            else:
                leaf.he = 1
                cur_node = leaf.parent
                while cur_node is not None:
                    cur_node.min_steps_to_correct = min(
                        [n.min_steps_to_correct for n in cur_node.children.values()]) + 1
                    cur_node.he = 1
                    cur_node = cur_node.parent
        for leaf in end_leaf_nodes:
            if leaf.min_steps_to_correct > 1:
                cur_node = leaf.parent
                while cur_node is not None and cur_node.min_steps_to_correct == 1024:
                    cur_node = cur_node.parent
                if cur_node is None:
                    continue
                else:
                    m = cur_node.min_steps_to_correct
                    cur_node = leaf
                    while cur_node.min_steps_to_correct == 1024:
                        cur_node.min_steps_to_correct = m
                        cur_node = cur_node.parent
            else:
                continue
        value_samples = self.get_all_value_samples_vm()
        return value_samples

    def get_all_value_samples_prm(self):
        # Retrieves all value samples from the subtree, considering the final route.
        full_value_samples = []
        if self.on_final_route:
            full_value_samples.append({'steps': self.y, 'value': self.he})
            if self.isFullyExpanded:
                for child in self.children.values():
                    if child.on_final_route:
                        sub_samples = child.get_all_value_samples_prm()
                        full_value_samples.extend(sub_samples)
            return full_value_samples
        else:
            return []

    def get_full_value_samples_prm(self, end_leaf_nodes):
        # Retrieves full value samples from the subtree, considering end leaf nodes and the final route.
        for leaf in end_leaf_nodes:
            cur_node = leaf.parent
            while cur_node is not None:
                cur_node.on_final_route = True
                cur_node = cur_node.parent
        for leaf in end_leaf_nodes:
            cur_node = leaf.parent
            while cur_node is not None:
                cur_node.he = max([n.he for n in cur_node.children.values() if n.on_final_route])
                cur_node = cur_node.parent
        value_samples = self.get_all_value_samples_prm()
        return value_samples