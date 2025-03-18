import numpy as np

class Cube:

    def __inti__(self):
        # create a 3x3x3 cube with each face represented by a 3x3 matrix
        self.faces = np.ones((3, 3, 6), dtype=int)

        # Initialize each face with a unique number (color)
        for i in range(6):
            self.faces[:, :, i] *= i

    def __repr__(self):
        # Create a string representation of the cube
        cube_str = ""
        for i in range(6):
            cube_str += f"Face {i}:\n{self.faces[:, :, i]}\n\n"
        return cube_str
    
    def rotate_face(self, face_index):
        # Rotate a face of the cube 90 degrees clockwise
        face = self.faces[:, :, face_index]
        self.faces[:, :, face_index] = np.rot90(face, -1)

        # rotate the adjacent edges
        if face_index == 0:
            temp = self.faces[0, :, 1].copy()
            self.faces[0, :, 1] = self.faces[:, 2, 2]
            self.faces[:, 2, 2] = self.faces[2, :, 1][::-1]
            self.faces[2, :, 1] = self.faces[:, 0, 0][::-1]
            self.faces[:, 0, 0] = temp[::-1]
        elif face_index == 1:
            temp = self.faces[:, 0, 1].copy()
            self.faces[:, 0, 1] = self.faces[2, :, 0][::-1]
            self.faces[2, :, 0] = self.faces[:, 2, 1]
            self.faces[:, 2, 1] = self.faces[0, :, 2]
            self.faces[0, :, 2] = temp
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
        if direction == 'clockwise':
            # Rotate clockwise
            self.rotate_face(face_index)
        elif direction == 'counterclockwise':
            # Rotate counterclockwise by rotating three times clockwise
            self.rotate_face(face_index)
            self.rotate_face(face_index)
            self.rotate_face(face_index)
        elif direction == 'half':
            # Rotate half by rotating twice clockwise
            self.rotate_face(face_index)
            self.rotate_face(face_index)