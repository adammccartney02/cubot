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
            self.faces[0, :, 0] = self.faces[:, 0, 2]
            self.faces[:, 0, 2] = self.faces[2, :, 5]
            self.faces[2, :, 5] = self.faces[:, 2, 4][::-1]
            self.faces[:, 2, 4] = temp[::-1]
        elif face_index == 2:
            temp = self.faces[2, :, 1].copy()
            self.faces[2, :, 1] = self.faces[:, 0, 2][::-1]
            self.faces[:, 0, 2] = self.faces[0, :, 1]
            self.faces[0, :, 1] = self.faces[:, 2, 0]
            self.faces[:, 2, 0] = temp
        elif face_index == 3:
            temp = self.faces[:, 2, 1].copy()
            self.faces[:, 2, 1] = self.faces[0, :, 0]
            self.faces[0, :, 0] = self.faces[:, 0, 1][::-1]
            self.faces[:, 0, 1] = self.faces[2, :, 2][::-1]
            self.faces[2, :, 2] = temp[::-1]
        elif face_index == 4:
            temp = self.faces[1, :, 0].copy()
            self.faces[1, :, 0] = self.faces[2, :, 1][::-1]
            self.faces[2, :, 1] = self.faces[3, :, 0]
            self.faces[3, :, 0] = self.faces[0, :, 1]
            self.faces[0, :, 1] = temp[::-1]
        elif face_index == 5:
            temp = self.faces[1, :, 2].copy()
            self.faces[1, :, 2] = self.faces[0, :, 1]
            self.faces[0, :, 1] = self.faces[3, :, 2][::-1]
            self.faces[3, :, 2] = self.faces[2, :, 1][::-1]
            self.faces[2, :, 1] = temp
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