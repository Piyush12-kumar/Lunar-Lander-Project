import numpy as np

def policy_action(policy, observation):
    """
    Converts an 8-dimensional observation into an action using a linear mapping.
    
    Expects the policy to be a flattened 36-element vector:
      - The first 32 elements are reshaped into an 8x4 weight matrix.
      - The last 4 elements are the bias.
    
    Returns the discrete action (the index of the highest logit).
    """
    num_inputs = 8
    num_outputs = 4
    weight = policy[:num_inputs * num_outputs].reshape((num_inputs, num_outputs))
    bias = policy[num_inputs * num_outputs:]
    logits = np.dot(observation, weight) + bias
    return int(np.argmax(logits))