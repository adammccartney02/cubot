import numpy as np

class Cube:

    def __init__(self):
        # create a 3x3x3 cube with each face represented by a 3x3 matrix
        self.faces = np.ones((3, 3, 6), dtype=int)

        # Initialize each face with a unique number (color)
        for i in range(6):
            self.faces[:, :, i] *= i
    
    def __repr__(self):
        # Create a string representation of the cube
        cube_str = ""
        colors = ['White', 'Red', 'Green', 'Orange', 'Blue', 'Yellow'] 
        for i in range(6):
            cube_str += f"{colors[i]}:\n{self.faces[:, :, i]}\n\n"
        return cube_str
    
    def __str__(self):
        '''
         Y 
         R 
        BWG
         O
        '''
        def box(id):
            match id:
                case 0:
                    return "\033[47m   \033[0m"  # White
                case 1:
                    return "\033[41m   \033[0m"  # Red
                case 2:
                    return "\033[42m   \033[0m" # Green
                case 3:
                    return "\033[45m   \033[0m" # Orange its actually purple :(
                case 4:
                    return "\033[46m   \033[0m" # Blue
                case 5:
                    return "\033[43m   \033[0m" #"\033[33m \u25A0 \033[0m" # Yellow
                case _:
                    return " "
                
        display_str = \
        '           ' + box(self.faces[0, 0, 5]) + box(self.faces[0, 1, 5]) + box(self.faces[0, 2, 5]) + '\n' + \
        '           ' + box(self.faces[1, 0, 5]) + box(self.faces[1, 1, 5]) + box(self.faces[1, 2, 5]) + '\n' + \
        '           ' + box(self.faces[2, 0, 5]) + box(self.faces[2, 1, 5]) + box(self.faces[2, 2, 5]) + '\n' + \
        '\n' + \
        '           ' + box(self.faces[0, 0, 1]) + box(self.faces[0, 1, 1]) + box(self.faces[0, 2, 1]) + '\n' + \
        '           ' + box(self.faces[1, 0, 1]) + box(self.faces[1, 1, 1]) + box(self.faces[1, 2, 1]) + '\n' + \
        '           ' + box(self.faces[2, 0, 1]) + box(self.faces[2, 1, 1]) + box(self.faces[2, 2, 1]) + '\n' + \
        '\n' + \
        box(self.faces[0, 2, 4]) + box(self.faces[1, 2, 4]) + box(self.faces[2, 2, 4]) + '  ' + box(self.faces[0, 0, 0]) + box(self.faces[0, 1, 0]) + box(self.faces[0, 2, 0]) + '  ' + box(self.faces[2, 0, 2]) + box(self.faces[1, 0, 2]) + box(self.faces[0, 0, 2]) + '\n' + \
        box(self.faces[0, 1, 4]) + box(self.faces[1, 1, 4]) + box(self.faces[2, 1, 4]) + '  ' + box(self.faces[1, 0, 0]) + box(self.faces[1, 1, 0]) + box(self.faces[1, 2, 0]) + '  ' + box(self.faces[2, 1, 2]) + box(self.faces[1, 1, 2]) + box(self.faces[0, 1, 2]) + '\n' + \
        box(self.faces[0, 0, 4]) + box(self.faces[1, 0, 4]) + box(self.faces[2, 0, 4]) + '  ' + box(self.faces[2, 0, 0]) + box(self.faces[2, 1, 0]) + box(self.faces[2, 2, 0]) + '  ' + box(self.faces[2, 2, 2]) + box(self.faces[1, 2, 2]) + box(self.faces[0, 2, 2]) + '\n' + \
        '\n' + \
        '           ' + box(self.faces[2, 2, 3]) + box(self.faces[2, 1, 3]) + box(self.faces[2, 0, 3]) + '\n' + \
        '           ' + box(self.faces[1, 2, 3]) + box(self.faces[1, 1, 3]) + box(self.faces[1, 0, 3]) + '\n' + \
        '           ' + box(self.faces[0, 2, 3]) + box(self.faces[0, 1, 3]) + box(self.faces[0, 0, 3]) + '\n'

        return display_str
    
    def rotate_face(self, face_index):
        # Rotate a face of the cube 90 degrees clockwise
        face = self.faces[:, :, face_index]
        self.faces[:, :, face_index] = np.rot90(face, -1)

        # rotate the adjacent edges
        if face_index == 0:
            # Rotate the White face (0)
            temp = self.faces[2, :, 1].copy()
            self.faces[2, :, 1] = self.faces[2, :, 4]
            self.faces[2, :, 4] = self.faces[2, :, 3]
            self.faces[2, :, 3] = self.faces[2, :, 2]
            self.faces[2, :, 2] = temp
        elif face_index == 1:
            # Rotate the Red face (1)
            temp = self.faces[0, :, 0].copy()
            self.faces[0, :, 0] = self.faces[:, 0, 2][::-1]
            self.faces[:, 0, 2] = self.faces[2, :, 5]
            self.faces[2, :, 5] = self.faces[:, 2, 4][::-1]
            self.faces[:, 2, 4] = temp
        elif face_index == 2:
            # Rotate the Green face (2)
            temp = self.faces[:, 2, 0].copy()
            self.faces[:, 2, 0] = self.faces[:, 0, 3][::-1]
            self.faces[:, 0, 3] = self.faces[:, 2, 5][::-1]
            self.faces[:, 2, 5] = self.faces[:, 2, 1]
            self.faces[:, 2, 1] = temp
        elif face_index == 3:
            # Rotate the Orange face (3)
            temp = self.faces[2, :, 0].copy()
            self.faces[2, :, 0] = self.faces[:, 0, 4]
            self.faces[:, 0, 4] = self.faces[0, :, 5][::-1]
            self.faces[0, :, 5] = self.faces[:, 2, 2]
            self.faces[:, 2, 2] = temp[::-1]
        elif face_index == 4:
            # Rotate the Blue face (4)
            temp = self.faces[:, 0, 0].copy()
            self.faces[:, 0, 0] = self.faces[:, 0, 1]
            self.faces[:, 0, 1] = self.faces[:, 0, 5]
            self.faces[:, 0, 5] = self.faces[:, 2, 3][::-1]
            self.faces[:, 2, 3] = temp[::-1]
        elif face_index == 5:
            # Rotate the Yellow face (5)
            temp = self.faces[0, :, 1].copy()
            self.faces[0, :, 1] = self.faces[0, :, 2]
            self.faces[0, :, 2] = self.faces[0, :, 3]
            self.faces[0, :, 3] = self.faces[0, :, 4]
            self.faces[0, :, 4] = temp
        else:
            raise ValueError("Invalid face index. Must be between 0 and 5.")
    
    def __call__(self, face_index, direction):
        if direction == 'cw':
            # Rotate clockwise
            self.rotate_face(face_index)
        elif direction == 'cc':
            # Rotate counterclockwise by rotating three times clockwise
            self.rotate_face(face_index)
            self.rotate_face(face_index)
            self.rotate_face(face_index)
        elif direction == 'hf':
            # Rotate half by rotating twice clockwise
            self.rotate_face(face_index)
            self.rotate_face(face_index)
        else:
            raise ValueError("Invalid direction. Must be 'cw', 'cc', or 'hf'.")
        
    def flat_state(self):
        # flatten the cube into a 1D array
        flat_ints = self.faces.flatten()

        # remove the center pieces
        flat_ints = np.delete(flat_ints, [24, 25, 26, 27, 28, 29])

        # build a binary array for each color and concatenate them
        binary_array = []
        for color in range(6):
            binary_array.append((flat_ints == color).astype(int))
        
        return np.concatenate(binary_array)
    
    def copy(self):
        # Create a deep copy of the cube
        new_cube = Cube()
        new_cube.faces = self.faces.copy()
        return new_cube
    
    def __eq__(self, other:'Cube'):
        return np.array_equal(self.faces, other.faces)