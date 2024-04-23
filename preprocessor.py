import jax.numpy as jnp
import numpy as np


class Preprocessor:
    """
    Processes the data, turns into an array of [58, 2] arrays
    Methods:
    -normalize: normalizes a feature by the max value
    -pad_sequence: pads the course_vertices array
    -encode_input: turns the json object into a [58, 2] array
    -proprocess
    """

    def __init__(self, max_pos=32767, max_vel=84, seq_length=55):
        self.max_pos = max_pos
        self.max_vel = max_vel
        self.seq_length = seq_length

    def normalize(self, value, max_value):
        """ Normalize a feature by the maximum value. """
        return value / max_value

    def pad_sequence(self, sequence, pad_value=0.0):
        """ Pad the sequence to ensure it has a fixed length. """
        num_pads = self.seq_length - len(sequence)
        padding = [[pad_value, pad_value]] * num_pads
        padded_sequence = sequence + padding
        return padded_sequence[:self.seq_length]

    def one_hot_encoded(self, label_sequence):
        """
        One hot encodes a sequence of integers between 0 and 7 inclusive
        """
        label_sequence = np.array(label_sequence)
        one_hot = np.eye(8)[label_sequence]

        return one_hot

    def encode_input(self, pos, vel, dir, vertices):
        """
        converts the input into a padded array of shape [58, 2]
        """
        pos = [self.normalize(pos['x'], self.max_pos),
               self.normalize(pos['y'], self.max_pos)]
        vel = [self.normalize(vel['x'], self.max_vel),
               self.normalize(vel['y'], self.max_vel)]
        dir_encoded = [1, 0] if dir == 'right' else [0, 1]
        normalized_vertices = [
            [
                self.normalize(v['x'], self.max_pos),
                self.normalize(v['y'], self.max_pos)
            ]
            for v in vertices
        ]
        normalized_vertices = self.pad_sequence(normalized_vertices)
        input_sequence = [pos, vel, dir_encoded] + normalized_vertices
        input_sequence = self.pad_sequence(input_sequence)
        output = np.array(input_sequence)
        return output

    def prepend_sos(self, sequence, token):
        """
        Prepends a sos token to the sequence
        """
        sequence.insert(0, token)
        return sequence

    def preprocess(self, data):
        """ Preprocess each datapoint in the dataset """
        processed_data = []
        for item in data:
            inputs = item['inputs']
            pos = inputs['pos']
            vel = inputs['vel']
            dir = inputs['dir']
            vertices = inputs['course_vertices']
            model_input = self.encode_input(pos, vel, dir, vertices)
            processed_data.append(model_input)
        return data
