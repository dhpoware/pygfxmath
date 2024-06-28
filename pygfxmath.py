# pygfxmath.py
# Copyright (c) 2024 dhpoware. All Rights Reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
# OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.

"""
pygfxmath is a simple computer graphics math library for Python.
It is a port of the C++ dhpoware 3D math library (https://github.com/dhpoware/mathlib).

For further information on how this library works see:
    Fletcher Dunn and Ian Parbery, "3D Math Primer for Graphics and Game Development 2nd Edition", https://gamemath.com/
"""

import math

from numbers import Real

def close_enough(a, b):
    """Wrapper around the math.isclose() function using tolerances specific to this module."""
    return math.isclose(a, b, abs_tol=1e-10)


class Vector2(object):
    """
    A 2-component vector class that represents a row vector.
    This class can also be used to represent a 2D cartesian coordinate (aka 2D point).
    """
    def __init__(self, x=0.0, y=0.0):
        self.x = x
        self.y = y

    def __str__(self):
        return f"({self.x}, {self.y})"

    def __eq__(self, other):
        if not isinstance(other, Vector2):
            return NotImplemented
        return close_enough(self.x, other.x) and close_enough(self.y, other.y)

    def __ne__(self, other):
        if not isinstance(other, Vector2):
            return NotImplemented
        return not self == other

    def __neg__(self):
        return Vector2(-self.x, -self.y)

    def __mul__(self, other):
        if not isinstance(other, Real):
            return NotImplemented
        return Vector2(self.x * other, self.y * other)

    def __rmul__(self, other):
        return self * other

    def __add__(self, other):
        if not isinstance(other, Vector2):
            return NotImplemented
        return Vector2(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        if not isinstance(other, Vector2):
            return NotImplemented
        return Vector2(self.x - other.x, self.y - other.y)

    def __truediv__(self, other):
        if not isinstance(other, Real):
            return NotImplemented
        return Vector2(self.x / other, self.y / other)

    def magnitude(self):
        """Calculate the length of the vector."""
        return math.sqrt((self.x * self.x) + (self.y * self.y))

    def magnitudesq(self):
        """Calculate the squared length of the vector."""
        return (self.x * self.x) + (self.y * self.y)

    def inverse(self):
        """Calculate the inverse of the vector."""
        return Vector2(-self.x, -self.y)

    def normalize(self):
        """Normalize the vector to make it unit length."""
        invMag = 1.0 / self.magnitude();
        self.x *= invMag
        self.y *= invMag

    @staticmethod
    def distancesq(pt1, pt2):
        """Calculate the squared distance between points pt1 and pt2."""
        if not isinstance(pt1, Vector2) or not isinstance(pt2, Vector2):
            raise NotImplementedError
        return ((pt1.x - pt2.x) * (pt1.x - pt2.x)) + ((pt1.y - pt2.y) * (pt1.y - pt2.y))

    @staticmethod
    def distance(pt1, pt2):
        """Calculate the distance between points pt1 and pt2."""
        if not isinstance(pt1, Vector2) or not isinstance(pt2, Vector2):
            raise NotImplementedError
        return math.sqrt(Vector2.distancesq(pt1, pt2))

    @staticmethod
    def dot(p, q):
        """Calculate the dot (aka scalar product) of vectors p and q."""
        if not isinstance(p, Vector2) or not isinstance(q, Vector2):
            raise NotImplementedError
        return (p.x * q.x) + (p.y * q.y)

    @staticmethod
    def lerp(p, q, t):
        """Linearly interpolates from 'p' to 'q' as t varies from 0 to 1."""
        if not isinstance(p, Vector2) or not isinstance(q, Vector2) or not isinstance(t, Real):
            raise NotImplementedError
        return p + t * (q - p)

    @staticmethod
    def proj(p, q):
        """Calculate the projection of 'p' onto 'q'."""
        if not isinstance(p, Vector2) or not isinstance(q, Vector2):
            raise NotImplementedError
        length = q.magnitude()
        return (Vector2.dot(p, q) / (length * length)) * q

    @staticmethod
    def perp(p, q):
        """Calculate the component of 'p' perpendicular to 'q'."""
        if not isinstance(p, Vector2) or not isinstance(q, Vector2):
            raise NotImplementedError
        length = q.magnitude()
        return p - ((Vector2.dot(p, q) / (length * length)) * q)

    @staticmethod
    def reflect(i, n):
        """Calculate the reflection vector from entering ray direction 'i' and surface normal 'n'."""
        if not isinstance(i, Vector2) or not isinstance(n, Vector2):
            raise NotImplementedError
        return i - 2.0 * Vector2.proj(i, n)


class Vector3(object):
    """
    A 3-component vector class that represents a row vector.
    This class can also be used to represent a 3D cartesian coordinate (aka 3D point).
    """
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x = x
        self.y = y
        self.z = z

    def __str__(self):
        return f"({self.x}, {self.y}, {self.z})"

    def __eq__(self, other):
        if not isinstance(other, Vector3):
            return NotImplemented
        return close_enough(self.x, other.x) and close_enough(self.y, other.y) and close_enough(self.z, other.z)

    def __ne__(self, other):
        if not isinstance(other, Vector3):
            return NotImplemented
        return not self == other

    def __neg__(self):
        return Vector3(-self.x, -self.y, -self.z)

    def __mul__(self, other):
        if not isinstance(other, Real):
            return NotImplemented
        return Vector3(self.x * other, self.y * other, self.z * other)

    def __rmul__(self, other):
        return self * other

    def __add__(self, other):
        if not isinstance(other, Vector3):
            return NotImplemented
        return Vector3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other):
        if not isinstance(other, Vector3):
            return NotImplemented
        return Vector3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __truediv__(self, other):
        if not isinstance(other, Real):
            return NotImplemented
        return Vector3(self.x / other, self.y / other, self.z / other)

    def magnitude(self):
        """Calculate the length of the vector."""
        return math.sqrt((self.x * self.x) + (self.y * self.y) + (self.z * self.z))

    def magnitudesq(self):
        """Calculate the squared length of the vector."""
        return (self.x * self.x) + (self.y * self.y) + (self.z * self.z)

    def inverse(self):
        """Calculate the inverse of the vector."""
        return Vector3(-self.x, -self.y, -self.z)

    def normalize(self):
        """Normalize the vector to make it unit length."""
        invMag = 1.0 / self.magnitude();
        self.x *= invMag
        self.y *= invMag
        self.z *= invMag

    @staticmethod
    def cross(p, q):
        """Calculate the cross product between vectors 'p' and 'q'."""
        if not isinstance(p, Vector3) or not isinstance(q, Vector3):
            raise NotImplementedError
        return Vector3((p.y * q.z) - (p.z * q.y), (p.z * q.x) - (p.x * q.z), (p.x * q.y) - (p.y * q.x))

    @staticmethod
    def distancesq(pt1, pt2):
        """Calculate the squared distance between points pt1 and pt2."""
        if not isinstance(pt1, Vector3) or not isinstance(pt2, Vector3):
            raise NotImplementedError
        return ((pt1.x - pt2.x) * (pt1.x - pt2.x)) + ((pt1.y - pt2.y) * (pt1.y - pt2.y)) + ((pt1.z - pt2.z) * (pt1.z - pt2.z))

    @staticmethod
    def distance(pt1, pt2):
        """Calculate the distance between points pt1 and pt2."""
        if not isinstance(pt1, Vector3) or not isinstance(pt2, Vector3):
            raise NotImplementedError
        return math.sqrt(Vector3.distancesq(pt1, pt2))

    @staticmethod
    def dot(p, q):
        """Calculate the dot (aka scalar product) of vectors p and q."""
        if not isinstance(p, Vector3) or not isinstance(q, Vector3):
            raise NotImplementedError
        return (p.x * q.x) + (p.y * q.y) + (p.z * q.z)

    @staticmethod
    def lerp(p, q, t):
        """Linearly interpolates from 'p' to 'q' as t varies from 0 to 1."""
        if not isinstance(p, Vector3) or not isinstance(q, Vector3) or not isinstance(t, Real):
            raise NotImplementedError
        return p + t * (q - p)

    @staticmethod
    def proj(p, q):
        """Calculate the projection of 'p' onto 'q'."""
        if not isinstance(p, Vector3) or not isinstance(q, Vector3):
            raise NotImplementedError
        length =  q.magnitude()
        return (Vector3.dot(p, q) / (length * length)) * q

    @staticmethod
    def perp(p, q):
        """Calculate the component of 'p' perpendicular to 'q'."""
        if not isinstance(p, Vector3) or not isinstance(q, Vector3):
            raise NotImplementedError
        length = q.magnitude()
        return p - ((Vector3.dot(p, q) / (length * length)) * q)

    @staticmethod
    def reflect(i, n):
        """Calculate the reflection vector from entering ray direction 'i' and surface normal 'n'."""
        if not isinstance(i, Vector3) or not isinstance(n, Vector3):
            raise NotImplementedError
        return i - 2.0 * Vector3.proj(i, n)

    @staticmethod
    def orthogonalize2(v1, v2):
        """
        Perform Gram-Schmidt Orthogonalization on the 2 basis vectors (v1 and v2) to turn them into orthonormal basis vectors.
        The orthonormal basis vectors are returned as a 2-element tuple.
        """
        if not isinstance(v1, Vector3) or not isinstance(v2, Vector3):
            raise NotImplementedError
        x = v1
        y = v2 - Vector3.proj(v2, x)
        y.normalize()
        return (x, y)

    @staticmethod
    def orthogonalize3(v1, v2, v3):
        """
        Perform Gram-Schmidt Orthogonalization on the 3 basis vectors (v1, v2, and v3) to turn them into orthonormal basis vectors.
        The orthonormal basis vectors are returned as a 3-element tuple.
        """
        if not isinstance(v1, Vector3) or not isinstance(v2, Vector3) or not isinstance(v3, Vector3):
            raise NotImplementedError
        x = v1
        y = v2 - Vector3.proj(v2, x)
        y.normalize();
        z = v3 - Vector3.proj(v3, x) - Vector3.proj(v3, y)
        z.normalize()
        return (x, y, z)


class Vector4(object):
    """
    A 4-component row vector class that represents a point or vector in homogeneous coordinates.
    A point has a w component of 1, and a vector has a w component of 0.
    """
    def __init__(self, x=0.0, y=0.0, z=0.0, w=0.0):
        self.x = x
        self.y = y
        self.z = z
        self.w = w

    def __str__(self):
        return f"({self.x}, {self.y}, {self.z}, {self.w})"

    def __eq__(self, other):
        if not isinstance(other, Vector4):
            return NotImplemented
        return close_enough(self.x, other.x) and close_enough(self.y, other.y) and close_enough(self.z, other.z) and close_enough(self.w, other.w)

    def __ne__(self, other):
        if not isinstance(other, Vector4):
            return NotImplemented
        return not self == other

    def __neg__(self):
        return Vector4(-self.x, -self.y, -self.z, -self.w)

    def __mul__(self, other):
        if not isinstance(other, Real):
            return NotImplemented
        return Vector4(self.x * other, self.y * other, self.z * other, self.w * other)

    def __rmul__(self, other):
        return self * other

    def __add__(self, other):
        if not isinstance(other, Vector4):
            return NotImplemented
        return Vector4(self.x + other.x, self.y + other.y, self.z + other.z, self.w + other.w)

    def __sub__(self, other):
        if not isinstance(other, Vector4):
            return NotImplemented
        return Vector4(self.x - other.x, self.y - other.y, self.z - other.z, self.w - other.w)

    def __truediv__(self, other):
        if not isinstance(other, Real):
            return NotImplemented
        return Vector4(self.x / other, self.y / other, self.z / other, self.w / other)

    def magnitude(self):
        """Calculate the length of the vector."""
        return math.sqrt((self.x * self.x) + (self.y * self.y) + (self.z * self.z) + (self.w * self.w))

    def magnitudesq(self):
        """Calculate the squared length of the vector."""
        return (self.x * self.x) + (self.y * self.y) + (self.z * self.z) + (self.w * self.w)

    def inverse(self):
        """Calculate the inverse of the vector."""
        return Vector4(-self.x, -self.y, -self.z, -self.w)

    def normalize(self):
        """Normalize the vector to make it unit length."""
        invMag = 1.0 / self.magnitude()
        self.x *= invMag
        self.y *= invMag
        self.z *= invMag
        self.w *= invMag

    def to_vector3(self):
        return Vector3(self.x / self.w, self.y / self.w, self.z / self.w) if self.w != 0.0 else Vector3(self.x, self.y, self.z)

    @staticmethod
    def from_vector3(v, w):
        if not isinstance(v, Vector3) and not isinstance(w, Real):
            raise NotImplementedError
        return Vector4(v.x, v.y, v.z, w)

    @staticmethod
    def distancesq(pt1, pt2):
        """Calculate the squared distance between points pt1 and pt2."""
        if not isinstance(pt1, Vector4) or not isinstance(pt2, Vector4):
            raise NotImplementedError
        return ((pt1.x - pt2.x) * (pt1.x - pt2.x)) + ((pt1.y - pt2.y) * (pt1.y - pt2.y)) + ((pt1.z - pt2.z) * (pt1.z - pt2.z)) + ((pt1.w - pt2.w) * (pt1.w - pt2.w))

    @staticmethod
    def distance(pt1, pt2):
        """Calculate the distance between points pt1 and pt2."""
        if not isinstance(pt1, Vector4) or not isinstance(pt2, Vector4):
            raise NotImplementedError
        return math.sqrt(Vector4.distancesq(pt1, pt2))

    @staticmethod
    def dot(p, q):
        """Calculate the dot (aka scalar product) of vectors p and q."""
        if not isinstance(p, Vector4) or not isinstance(q, Vector4):
            raise NotImplementedError
        return (p.x * q.x) + (p.y * q.y) + (p.z * q.z) + (p.w * q.w)

    @staticmethod
    def lerp(p, q, t):
        """Linearly interpolates from 'p' to 'q' as t varies from 0 to 1."""
        if not isinstance(p, Vector4) or not isinstance(q, Vector4) or not isinstance(t, Real):
            raise NotImplementedError
        return p + t * (q - p)


class Matrix3(object):
    """
    A row-major 3x3 matrix class.
    Matrices are multiplied in a left to right order.
    Multiplies vectors to the left of the matrix.
    """
    def __init__(self, row1=[0.0, 0.0, 0.0], row2=[0.0, 0.0, 0.0], row3=[0.0, 0.0, 0.0]):
        self.mtx = [[0.0, 0.0, 0.0] for i in range(3)]
        self.mtx[0][0] = row1[0]
        self.mtx[0][1] = row1[1]
        self.mtx[0][2] = row1[2]
        self.mtx[1][0] = row2[0]
        self.mtx[1][1] = row2[1]
        self.mtx[1][2] = row2[2]
        self.mtx[2][0] = row3[0]
        self.mtx[2][1] = row3[1]
        self.mtx[2][2] = row3[2]

    def __str__(self):
        return f"{self.mtx}"

    def __eq__(self, other):
        if not isinstance(other, Matrix3):
            return NotImplemented
        return  close_enough(self.mtx[0][0], other.mtx[0][0]) \
            and close_enough(self.mtx[0][1], other.mtx[0][1]) \
            and close_enough(self.mtx[0][2], other.mtx[0][2]) \
            and close_enough(self.mtx[1][0], other.mtx[1][0]) \
            and close_enough(self.mtx[1][1], other.mtx[1][1]) \
            and close_enough(self.mtx[1][2], other.mtx[1][2]) \
            and close_enough(self.mtx[2][0], other.mtx[2][0]) \
            and close_enough(self.mtx[2][1], other.mtx[2][1]) \
            and close_enough(self.mtx[2][2], other.mtx[2][2])
    def __ne__(self, other):
        if not isinstance(other, Matrix3):
            return NotImplemented
        return not self == other

    def __neg__(self):
        return Matrix3(
            [-self.mtx[0][0], -self.mtx[0][1], -self.mtx[0][2]],
            [-self.mtx[1][0], -self.mtx[1][1], -self.mtx[1][2]],
            [-self.mtx[2][0], -self.mtx[2][1], -self.mtx[2][2]]
        )

    def __mul__(self, other):
        if isinstance(other, Real):
            return Matrix3(
                [self.mtx[0][0] * other, self.mtx[0][1] * other, self.mtx[0][2] * other],
                [self.mtx[1][0] * other, self.mtx[1][1] * other, self.mtx[1][2] * other],
                [self.mtx[2][0] * other, self.mtx[2][1] * other, self.mtx[2][2] * other]
            )
        elif isinstance(other, Matrix3):
            return Matrix3(
                [(self.mtx[0][0] * other.mtx[0][0]) + (self.mtx[0][1] * other.mtx[1][0]) + (self.mtx[0][2] * other.mtx[2][0]),
                 (self.mtx[0][0] * other.mtx[0][1]) + (self.mtx[0][1] * other.mtx[1][1]) + (self.mtx[0][2] * other.mtx[2][1]),
                 (self.mtx[0][0] * other.mtx[0][2]) + (self.mtx[0][1] * other.mtx[1][2]) + (self.mtx[0][2] * other.mtx[2][2])],

                [(self.mtx[1][0] * other.mtx[0][0]) + (self.mtx[1][1] * other.mtx[1][0]) + (self.mtx[1][2] * other.mtx[2][0]),
                 (self.mtx[1][0] * other.mtx[0][1]) + (self.mtx[1][1] * other.mtx[1][1]) + (self.mtx[1][2] * other.mtx[2][1]),
                 (self.mtx[1][0] * other.mtx[0][2]) + (self.mtx[1][1] * other.mtx[1][2]) + (self.mtx[1][2] * other.mtx[2][2])],

                [(self.mtx[2][0] * other.mtx[0][0]) + (self.mtx[2][1] * other.mtx[1][0]) + (self.mtx[2][2] * other.mtx[2][0]),
                 (self.mtx[2][0] * other.mtx[0][1]) + (self.mtx[2][1] * other.mtx[1][1]) + (self.mtx[2][2] * other.mtx[2][1]),
                 (self.mtx[2][0] * other.mtx[0][2]) + (self.mtx[2][1] * other.mtx[1][2]) + (self.mtx[2][2] * other.mtx[2][2])]
            )
        elif isinstance(other, Vector3):
            return Vector3(
                (other.x * self.mtx[0][0]) + (other.y * self.mtx[1][0]) + (other.z * self.mtx[2][0]),
                (other.x * self.mtx[0][1]) + (other.y * self.mtx[1][1]) + (other.z * self.mtx[2][1]),
                (other.x * self.mtx[0][2]) + (other.y * self.mtx[1][2]) + (other.z * self.mtx[2][2])
            )
        else:
            return NotImplemented

    def __rmul__(self, other):
        return self * other

    def __add__(self, other):
        if not isinstance(other, Matrix3):
            return NotImplemented
        return Matrix3(
            [self.mtx[0][0] + other.mtx[0][0], self.mtx[0][1] + other.mtx[0][1], self.mtx[0][2] + other.mtx[0][2]],
            [self.mtx[1][0] + other.mtx[1][0], self.mtx[1][1] + other.mtx[1][1], self.mtx[1][2] + other.mtx[1][2]],
            [self.mtx[2][0] + other.mtx[2][0], self.mtx[2][1] + other.mtx[2][1], self.mtx[2][2] + other.mtx[2][2]]
        )

    def __sub__(self, other):
        if not isinstance(other, Matrix3):
            return NotImplemented
        return Matrix3(
            [self.mtx[0][0] - other.mtx[0][0], self.mtx[0][1] - other.mtx[0][1], self.mtx[0][2] - other.mtx[0][2]],
            [self.mtx[1][0] - other.mtx[1][0], self.mtx[1][1] - other.mtx[1][1], self.mtx[1][2] - other.mtx[1][2]],
            [self.mtx[2][0] - other.mtx[2][0], self.mtx[2][1] - other.mtx[2][1], self.mtx[2][2] - other.mtx[2][2]]
        )

    def __truediv__(self, other):
        if not isinstance(other, Real):
            return NotImplemented
        return Matrix3(
            [self.mtx[0][0] / other, self.mtx[0][1] / other, self.mtx[0][2] / other],
            [self.mtx[1][0] / other, self.mtx[1][1] / other, self.mtx[1][2] / other],
            [self.mtx[2][0] / other, self.mtx[2][1] / other, self.mtx[2][2] / other]
        )

    def determinant(self):
        """Calculate the determinant of the matrix."""
        return (self.mtx[0][0] * (self.mtx[1][1] * self.mtx[2][2] - self.mtx[1][2] * self.mtx[2][1])) \
             - (self.mtx[0][1] * (self.mtx[1][0] * self.mtx[2][2] - self.mtx[1][2] * self.mtx[2][0])) \
             + (self.mtx[0][2] * (self.mtx[1][0] * self.mtx[2][1] - self.mtx[1][1] * self.mtx[2][0]))

    def from_axes(self, x_axis, y_axis, z_axis):
        """Create a rotation matrix from 3 basis vectors (x_axis, y_axis, and z_axis)."""
        if not isinstance(x, Vector3) or not isinstance(y, Vector3) or not isinstance(z, Vector3):
            raise NotImplementedError
        self.mtx[0][0] = x_axis.x
        self.mtx[0][1] = x_axis.y
        self.mtx[0][2] = x_axis.z
        self.mtx[1][0] = y_axis.x
        self.mtx[1][1] = y_axis.y
        self.mtx[1][2] = y_axis.z
        self.mtx[2][0] = z_axis.x
        self.mtx[2][1] = z_axis.y
        self.mtx[2][2] = z_axis.z

    def from_axes_transposed(self, x_axis, y_axis, z_axis):
        """Create a transposed rotation matrix from 3 basis vectors (x_axis, y_axis, and z_axis)."""
        if not isinstance(x, Vector3) or not isinstance(y, Vector3) or not isinstance(z, Vector3):
            raise NotImplementedError
        self.mtx[0][0] = x_axis.x
        self.mtx[0][1] = y_axis.x
        self.mtx[0][2] = z_axis.x
        self.mtx[1][0] = x_axis.y
        self.mtx[1][1] = y_axis.y
        self.mtx[1][2] = z_axis.y
        self.mtx[2][0] = x_axis.z
        self.mtx[2][1] = y_axis.z
        self.mtx[2][2] = z_axis.z

    def from_heading_pitch_roll(self, heading_degrees, pitch_degrees, roll_degrees):
        """
        Create a rotation matrix based on a Euler Transform.
        The popular NASA standard airplane convention of heading-pitch-roll (i.e., RzRxRy) is used here.
        """
        if not isinstance(heading_degrees, Real) or not isinstance(pitch_degrees, Real) or not isinstance(roll_degrees, Real):
            raise NotImplementedError

        heading = math.radians(heading_degrees)
        pitch = math.radians(pitch_degrees)
        roll = math.radians(roll_degrees)

        cos_heading = math.cos(heading)
        cos_pitch = math.cos(pitch)
        cos_roll = math.cos(roll)
        sin_heading = math.sin(heading)
        sin_pitch = math.sin(pitch)
        sin_roll = math.sin(roll)

        self.mtx[0][0] = cos_roll * cos_heading - sin_roll * sin_pitch * sin_heading
        self.mtx[0][1] = sin_roll * cos_heading + cos_roll * sin_pitch * sin_heading
        self.mtx[0][2] = -cos_pitch * sin_heading
        self.mtx[1][0] = -sin_roll * cos_pitch
        self.mtx[1][1] = cos_roll * cos_pitch
        self.mtx[1][2] = sin_pitch
        self.mtx[2][0] = cos_roll * sin_heading + sin_roll * sin_pitch * cos_heading
        self.mtx[2][1] = sin_roll * sin_heading - cos_roll * sin_pitch * cos_heading
        self.mtx[2][2] = cos_pitch * cos_heading

    def inverse(self):
        """Calculate the inverse of the matrix. If the inverse doesn't exist, the identity matrix is returned instead."""
        result = Matrix3()
        d = self.determinant()

        if close_enough(0.0, d):
            result.mtx[0][0] = 1.0
            result.mtx[0][1] = 0.0
            result.mtx[0][2] = 0.0
            result.mtx[1][0] = 0.0
            result.mtx[1][1] = 1.0
            result.mtx[1][2] = 0.0
            result.mtx[2][0] = 0.0
            result.mtx[2][1] = 0.0
            result.mtx[2][2] = 1.0
        else:
            d = 1.0 / d
            result.mtx[0][0] = d * (self.mtx[1][1] * self.mtx[2][2] - self.mtx[1][2] * self.mtx[2][1])
            result.mtx[0][1] = d * (self.mtx[0][2] * self.mtx[2][1] - self.mtx[0][1] * self.mtx[2][2])
            result.mtx[0][2] = d * (self.mtx[0][1] * self.mtx[1][2] - self.mtx[0][2] * self.mtx[1][1])
            result.mtx[1][0] = d * (self.mtx[1][2] * self.mtx[2][0] - self.mtx[1][0] * self.mtx[2][2])
            result.mtx[1][1] = d * (self.mtx[0][0] * self.mtx[2][2] - self.mtx[0][2] * self.mtx[2][0])
            result.mtx[1][2] = d * (self.mtx[0][2] * self.mtx[1][0] - self.mtx[0][0] * self.mtx[1][2])
            result.mtx[2][0] = d * (self.mtx[1][0] * self.mtx[2][1] - self.mtx[1][1] * self.mtx[2][0])
            result.mtx[2][1] = d * (self.mtx[0][1] * self.mtx[2][0] - self.mtx[0][0] * self.mtx[2][1])
            result.mtx[2][2] = d * (self.mtx[0][0] * self.mtx[1][1] - self.mtx[0][1] * self.mtx[1][0])

        return result

    def orient(self, from_vector, to_vector):
        """
        Create an orientation matrix that will rotate the vector 'from_vector' into the vector 'to_vector'.
        For this method to work correctly, both of these vectors must be unit length.

        The algorithm used is from:
            Tomas Moller and John F. Hughes, "Efficiently building a matrix to rotate one vector to another",
            Journal of Graphics Tools, 4(4):1-4, 1999.
        """
        if not isinstance(from_vector, Vector3) or not isinstance(to_vector, Vector3):
            raise NotImplementedError

        e = Vector3.dot(from_vector, to_vector)

        if close_enough(e, 1.0):
            # Special case where 'from_vector' is equal to 'to_vector'.
            # In other words, the angle between vector 'from_vector' and vector 'to_vector' is zero degrees.
            # In this case just load the identity matrix.
            self.mtx[0][0] = 1.0
            self.mtx[0][1] = 0.0
            self.mtx[0][2] = 0.0
            self.mtx[1][0] = 0.0
            self.mtx[1][1] = 1.0
            self.mtx[1][2] = 0.0
            self.mtx[2][0] = 0.0
            self.mtx[2][1] = 0.0
            self.mtx[2][2] = 1.0
        elif close_enough(e, -1.0):
            # Special case where 'from_vector' is directly opposite to 'to_vector'.
            # In other words, the angle between vector 'from_vector' and vector 'to_vector' is 180 degrees.
            # In this case, the following matrix is used:
            #
            # Let:
            #   F = from_vector
            #   S = vector perpendicular to F
            #   U = S X F
            #
            # We want to rotate from (F, U, S) to (-F, U, -S)
            #
            # | -FxFx+UxUx-SxSx  -FxFy+UxUy-SxSy  -FxFz+UxUz-SxSz |
            # | -FxFy+UxUy-SxSy  -FyFy+UyUy-SySy  -FyFz+UyUz-SySz |
            # | -FxFz+UxUz-SxSz  -FyFz+UyUz-SySz  -FzFz+UzUz-SzSz |
            side = Vector3(0.0, from_vector.z, -from_vector.y)

            if close_enough(Vector3.dot(side, side), 0.0):
                side.x = -from_vector.z
                side.y = 0.0
                side.z = from_vector.x

            side.normalize()

            up = Vector3.cross(side, from_vector)
            up.normalize()

            self.mtx[0][0] = -(from_vector.x * from_vector.x) + (up.x * up.x) - (side.x * side.x)
            self.mtx[0][1] = -(from_vector.x * from_vector.y) + (up.x * up.y) - (side.x * side.y)
            self.mtx[0][2] = -(from_vector.x * from_vector.z) + (up.x * up.z) - (side.x * side.z)
            self.mtx[1][0] = -(from_vector.x * from_vector.y) + (up.x * up.y) - (side.x * side.y)
            self.mtx[1][1] = -(from_vector.y * from_vector.y) + (up.y * up.y) - (side.y * side.y)
            self.mtx[1][2] = -(from_vector.y * from_vector.z) + (up.y * up.z) - (side.y * side.z)
            self.mtx[2][0] = -(from_vector.x * from_vector.z) + (up.x * up.z) - (side.x * side.z)
            self.mtx[2][1] = -(from_vector.y * from_vector.z) + (up.y * up.z) - (side.y * side.z)
            self.mtx[2][2] = -(from_vector.z * from_vector.z) + (up.z * up.z) - (side.z * side.z)
        else:
            # This is the most common case. Creates the rotation matrix:
            #
            #                             | E + HVx^2   HVxVy + Vz  HVxVz - Vy |
            # R(from_vector, to_vector) = | HVxVy - Vz  E + HVy^2   HVxVz + Vx |
            #                             | HVxVz + Vy  HVyVz - Vx  E + HVz^2  |
            #
            # where,
            #   V = from_vector.cross(to_vector)
            #   E = from_vector.dot(to_vector)
            #   H = (1 - E) / V.dot(V)
            v = Vector3.cross(from_vector, to_vector)
            v.normalize()

            h = (1.0 - e) / Vector3.dot(v, v)

            self.mtx[0][0] = e + h * v.x * v.x
            self.mtx[0][1] = h * v.x * v.y + v.z
            self.mtx[0][2] = h * v.x * v.z - v.y
            self.mtx[1][0] = h * v.x * v.y - v.z
            self.mtx[1][1] = e + h * v.y * v.y
            self.mtx[1][2] = h * v.x * v.z + v.x
            self.mtx[2][0] = h * v.x * v.z + v.y
            self.mtx[2][1] = h * v.y * v.z - v.x
            self.mtx[2][2] = e + h * v.z * v.z

    def rotate(self, axis, degrees):
        """
        Create a rotation matrix about the specified axis.
        The axis must be a unit vector and the angle must be in degrees.

        Let u = axis of rotation = (x, y, z)

                    | x^2(1 - c) + c  xy(1 - c) + zs  xz(1 - c) - ys |
        Ru(angle) = | yx(1 - c) - zs  y^2(1 - c) + c  yz(1 - c) + xs |
                    | zx(1 - c) - ys  zy(1 - c) - xs  z^2(1 - c) + c |

        where
            c = cos(angle)
            s = sin(angle)
        """
        if not isinstance(axis, Vector3) or not isinstance(degrees, Real):
            raise NotImplementedError

        angle = math.radians(degrees)

        x = axis.x
        y = axis.y
        z = axis.z
        c = math.cos(angle)
        s = math.sin(angle)

        self.mtx[0][0] = (x * x) * (1.0 - c) + c
        self.mtx[0][1] = (x * y) * (1.0 - c) + (z * s)
        self.mtx[0][2] = (x * z) * (1.0 - c) - (y * s)
        self.mtx[1][0] = (y * x) * (1.0 - c) - (z * s)
        self.mtx[1][1] = (y * y) * (1.0 - c) + c
        self.mtx[1][2] = (y * z) * (1.0 - c) + (x * s)
        self.mtx[2][0] = (z * x) * (1.0 - c) + (y * s)
        self.mtx[2][1] = (z * y) * (1.0 - c) - (x * s)
        self.mtx[2][2] = (z * z) * (1.0 - c) + c

    def scale(self, sx, sy, sz):
        """
        Create a scaling matrix.

                        | sx   0    0  |
        S(sx, sy, sz) = | 0    sy   0  |
                        | 0    0    sz |
        """
        self.mtx[0][0] = sx
        self.mtx[0][1] = 0.0
        self.mtx[0][2] = 0.0
        self.mtx[1][0] = 0.0
        self.mtx[1][1] = sy
        self.mtx[1][2] = 0.0
        self.mtx[2][0] = 0.0
        self.mtx[2][1] = 0.0
        self.mtx[2][2] = sz

    def to_axes(self):
        """
        Extract the local x, y, and z axes from the matrix.
        The x, y, and z axes are returned in a 3-element tuple of Vector3 objects in that order.
        This only makes sense for rotation matrices.
        Calling this method on a matrix that isn't a rotation matrix will lead to undefined behavior.
        """
        x_axis = Vector3(self.mtx[0][0], self.mtx[0][1], self.mtx[0][2])
        y_axis = Vector3(self.mtx[1][0], self.mtx[1][1], self.mtx[1][2])
        z_axis = Vector3(self.mtx[2][0], self.mtx[2][1], self.mtx[2][2])
        return (x_axis, y_axis, z_axis)

    def to_axes_transposed(self):
        """
        Extract the local x, y, and z axes from the transpose of the matrix.
        The x, y, and z axes are returned in a 3-element tuple of Vector3 objects in that order.
        This only makes sense for rotation matrices.
        Calling this method on a matrix that isn't a rotation matrix will lead to undefined behavior.
        """
        x_axis = Vector3(self.mtx[0][0], self.mtx[1][0], self.mtx[2][0])
        y_axis = Vector3(self.mtx[0][1], self.mtx[1][1], self.mtx[2][1])
        z_axis = Vector3(self.mtx[0][2], self.mtx[1][2], self.mtx[2][2])
        return (x_axis, y_axis, z_axis)

    def to_heading_pitch_roll(self):
        """
        Extract the Euler angles from a rotation matrix.
        The returned angles are in degrees.
        This method might suffer from numerical imprecision for ill defined rotation matrices.
        This function only works for rotation matrices constructed using the popular NASA standard airplane convention of heading-pitch-roll (i.e., RzRxRy).

        The algorithm used is from:
            David Eberly, "Euler Angle Formulas", Geometric Tools web site, http://www.geometrictools.com/Documentation/EulerAngles.pdf.

        The heading, pitch, and roll angles are returned in a 3-element tuple in that order.
        """
        theta_x = math.asin(self.mtx[1][2])
        theta_y = 0.0
        theta_z = 0.0
        half_pi = math.pi * 0.5

        if theta_x < half_pi:
            if theta_x > -half_pi:
                theta_z = math.atan2(-self.mtx[1][0], self.mtx[1][1])
                theta_y = math.atan2(-self.mtx[0][2], self.mtx[2][2])
            else:
                # Not a unique solution.
                theta_z = -math.atan2(self.mtx[2][0], self.mtx[0][0])
                theta_y = 0.0
        else:
            # Not a unique solution.
            theta_z = math.atan2(self.mtx[2][0], self.mtx[0][0])
            theta_y = 0.0

        heading = math.degrees(theta_y)
        pitch = math.degrees(theta_x)
        roll = math.degrees(theta_z)
        return (heading, pitch, roll)

    def transpose(self):
        """Return the transpose of the matrix."""
        return Matrix3(
            [self.mtx[0][0], self.mtx[1][0], self.mtx[2][0]],
            [self.mtx[0][1], self.mtx[1][1], self.mtx[2][1]],
            [self.mtx[0][2], self.mtx[1][2], self.mtx[2][2]]
        )

    @staticmethod
    def create_from_axes(x_axis, y_axis, z_axis):
        """Create a rotation matrix using the 3 provided basis vectors."""
        result = Matrix3()
        result.from_axes(x_axis, y_axis, z_axis)
        return result

    @staticmethod
    def create_from_axes_transposed(x_axis, y_axis, z_axis):
        """Create a transposed rotation matrix using the 3 provided basis vectors."""
        result = Matrix3()
        result.from_axes_transposed(x_axis, y_axis, z_axis)
        return result

    @staticmethod
    def create_from_heading_pitch_roll(heading_degrees, pitch_degrees, roll_degrees):
        """Create a rotation matrix based on a Euler Transform."""
        result = Matrix3()
        result.from_heading_pitch_roll(heading_degrees, pitch_degrees, roll_degrees)
        return result

    @staticmethod
    def create_orient(from_vector, to_vector):
        """Create an orientation matrix that will rotate the vector 'from_vector' into the vector 'to_vector'."""
        result = Matrix3()
        result.orient(from_vector, to_vector)
        return result

    @staticmethod
    def create_rotate(axis, degrees):
        """Create a rotation matrix about the specified axis. The axis must be a unit vector and the angle must be in degrees."""
        result = Matrix3()
        result.rotate(axis, degrees)
        return result

    @staticmethod
    def create_scale(sx, sy, sz):
        """Create a scaling matrix."""
        result = Matrix3()
        result.scale(sx, sy, sz)
        return result

    @staticmethod
    def identity():
        """Create an identity matrix."""
        return Matrix3([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0])


class Matrix4(object):
    """
    A homogeneous row-major 4x4 matrix class.
    Matrices are multiplied in a left to right order.
    Multiplies vectors to the left of the matrix.
    """
    def __init__(self, row1=[0.0, 0.0, 0.0, 0.0], row2=[0.0, 0.0, 0.0, 0.0], row3=[0.0, 0.0, 0.0, 0.0], row4=[0.0, 0.0, 0.0, 0.0]):
        self.mtx = [[0.0, 0.0, 0.0, 0.0] for i in range(4)]
        self.mtx[0][0] = row1[0]
        self.mtx[0][1] = row1[1]
        self.mtx[0][2] = row1[2]
        self.mtx[0][3] = row1[3]
        self.mtx[1][0] = row2[0]
        self.mtx[1][1] = row2[1]
        self.mtx[1][2] = row2[2]
        self.mtx[1][3] = row2[3]
        self.mtx[2][0] = row3[0]
        self.mtx[2][1] = row3[1]
        self.mtx[2][2] = row3[2]
        self.mtx[2][3] = row3[3]
        self.mtx[3][0] = row4[0]
        self.mtx[3][1] = row4[1]
        self.mtx[3][2] = row4[2]
        self.mtx[3][3] = row4[3]

    def __str__(self):
        return f"{self.mtx}"

    def __eq__(self, other):
        if not isinstance(other, Matrix4):
            return NotImplemented
        return  close_enough(self.mtx[0][0], other.mtx[0][0]) \
            and close_enough(self.mtx[0][1], other.mtx[0][1]) \
            and close_enough(self.mtx[0][2], other.mtx[0][2]) \
            and close_enough(self.mtx[0][3], other.mtx[0][3]) \
            and close_enough(self.mtx[1][0], other.mtx[1][0]) \
            and close_enough(self.mtx[1][1], other.mtx[1][1]) \
            and close_enough(self.mtx[1][2], other.mtx[1][2]) \
            and close_enough(self.mtx[1][3], other.mtx[1][3]) \
            and close_enough(self.mtx[2][0], other.mtx[2][0]) \
            and close_enough(self.mtx[2][1], other.mtx[2][1]) \
            and close_enough(self.mtx[2][2], other.mtx[2][2]) \
            and close_enough(self.mtx[2][3], other.mtx[2][3]) \
            and close_enough(self.mtx[3][0], other.mtx[3][0]) \
            and close_enough(self.mtx[3][1], other.mtx[3][1]) \
            and close_enough(self.mtx[3][2], other.mtx[3][2]) \
            and close_enough(self.mtx[3][3], other.mtx[3][3])

    def __ne__(self, other):
        if not isinstance(other, Matrix4):
            return NotImplemented
        return not self == other

    def __neg__(self):
        return Matrix4(
            [-self.mtx[0][0], -self.mtx[0][1], -self.mtx[0][2], -self.mtx[0][3]],
            [-self.mtx[1][0], -self.mtx[1][1], -self.mtx[1][2], -self.mtx[1][3]],
            [-self.mtx[2][0], -self.mtx[2][1], -self.mtx[2][2], -self.mtx[2][3]],
            [-self.mtx[3][0], -self.mtx[3][1], -self.mtx[3][2], -self.mtx[3][3]]
        )

    def __mul__(self, other):
        if isinstance(other, Real):
            return Matrix4(
                [self.mtx[0][0] * other, self.mtx[0][1] * other, self.mtx[0][2] * other, self.mtx[0][3] * other],
                [self.mtx[1][0] * other, self.mtx[1][1] * other, self.mtx[1][2] * other, self.mtx[1][3] * other],
                [self.mtx[2][0] * other, self.mtx[2][1] * other, self.mtx[2][2] * other, self.mtx[2][3] * other],
                [self.mtx[3][0] * other, self.mtx[3][1] * other, self.mtx[3][2] * other, self.mtx[3][3] * other]
            )
        elif isinstance(other, Matrix4):
            return Matrix4(
                [(self.mtx[0][0] * other.mtx[0][0]) + (self.mtx[0][1] * other.mtx[1][0]) + (self.mtx[0][2] * other.mtx[2][0]) + (self.mtx[0][3] * other.mtx[3][0]),
                 (self.mtx[0][0] * other.mtx[0][1]) + (self.mtx[0][1] * other.mtx[1][1]) + (self.mtx[0][2] * other.mtx[2][1]) + (self.mtx[0][3] * other.mtx[3][1]),
                 (self.mtx[0][0] * other.mtx[0][2]) + (self.mtx[0][1] * other.mtx[1][2]) + (self.mtx[0][2] * other.mtx[2][2]) + (self.mtx[0][3] * other.mtx[3][2]),
                 (self.mtx[0][0] * other.mtx[0][3]) + (self.mtx[0][1] * other.mtx[1][3]) + (self.mtx[0][2] * other.mtx[2][3]) + (self.mtx[0][3] * other.mtx[3][3])],

                [(self.mtx[1][0] * other.mtx[0][0]) + (self.mtx[1][1] * other.mtx[1][0]) + (self.mtx[1][2] * other.mtx[2][0]) + (self.mtx[1][3] * other.mtx[3][0]),
                 (self.mtx[1][0] * other.mtx[0][1]) + (self.mtx[1][1] * other.mtx[1][1]) + (self.mtx[1][2] * other.mtx[2][1]) + (self.mtx[1][3] * other.mtx[3][1]),
                 (self.mtx[1][0] * other.mtx[0][2]) + (self.mtx[1][1] * other.mtx[1][2]) + (self.mtx[1][2] * other.mtx[2][2]) + (self.mtx[1][3] * other.mtx[3][2]),
                 (self.mtx[1][0] * other.mtx[0][3]) + (self.mtx[1][1] * other.mtx[1][3]) + (self.mtx[1][2] * other.mtx[2][3]) + (self.mtx[1][3] * other.mtx[3][3])],

                [(self.mtx[2][0] * other.mtx[0][0]) + (self.mtx[2][1] * other.mtx[1][0]) + (self.mtx[2][2] * other.mtx[2][0]) + (self.mtx[2][3] * other.mtx[3][0]),
                 (self.mtx[2][0] * other.mtx[0][1]) + (self.mtx[2][1] * other.mtx[1][1]) + (self.mtx[2][2] * other.mtx[2][1]) + (self.mtx[2][3] * other.mtx[3][1]),
                 (self.mtx[2][0] * other.mtx[0][2]) + (self.mtx[2][1] * other.mtx[1][2]) + (self.mtx[2][2] * other.mtx[2][2]) + (self.mtx[2][3] * other.mtx[3][2]),
                 (self.mtx[2][0] * other.mtx[0][3]) + (self.mtx[2][1] * other.mtx[1][3]) + (self.mtx[2][2] * other.mtx[2][3]) + (self.mtx[2][3] * other.mtx[3][3])],

                [(self.mtx[3][0] * other.mtx[0][0]) + (self.mtx[3][1] * other.mtx[1][0]) + (self.mtx[3][2] * other.mtx[2][0]) + (self.mtx[3][3] * other.mtx[3][0]),
                 (self.mtx[3][0] * other.mtx[0][1]) + (self.mtx[3][1] * other.mtx[1][1]) + (self.mtx[3][2] * other.mtx[2][1]) + (self.mtx[3][3] * other.mtx[3][1]),
                 (self.mtx[3][0] * other.mtx[0][2]) + (self.mtx[3][1] * other.mtx[1][2]) + (self.mtx[3][2] * other.mtx[2][2]) + (self.mtx[3][3] * other.mtx[3][2]),
                 (self.mtx[3][0] * other.mtx[0][3]) + (self.mtx[3][1] * other.mtx[1][3]) + (self.mtx[3][2] * other.mtx[2][3]) + (self.mtx[3][3] * other.mtx[3][3])]
            )
        elif isinstance(other, Vector4):
            return Vector4(
                (other.x * self.mtx[0][0]) + (other.y * self.mtx[1][0]) + (other.z * self.mtx[2][0]) + (other.w * self.mtx[3][0]),
                (other.x * self.mtx[0][1]) + (other.y * self.mtx[1][1]) + (other.z * self.mtx[2][1]) + (other.w * self.mtx[3][1]),
                (other.x * self.mtx[0][2]) + (other.y * self.mtx[1][2]) + (other.z * self.mtx[2][2]) + (other.w * self.mtx[3][2]),
                (other.x * self.mtx[0][3]) + (other.y * self.mtx[1][3]) + (other.z * self.mtx[2][3]) + (other.w * self.mtx[3][3])
            )
        else:
            return NotImplemented

    def __rmul__(self, other):
        return self * other

    def __add__(self, other):
        if not isinstance(other, Matrix4):
            return NotImplemented
        return Matrix4(
            [self.mtx[0][0] + other.mtx[0][0], self.mtx[0][1] + other.mtx[0][1], self.mtx[0][2] + other.mtx[0][2], self.mtx[0][3] + other.mtx[0][3]],
            [self.mtx[1][0] + other.mtx[1][0], self.mtx[1][1] + other.mtx[1][1], self.mtx[1][2] + other.mtx[1][2], self.mtx[1][3] + other.mtx[1][3]],
            [self.mtx[2][0] + other.mtx[2][0], self.mtx[2][1] + other.mtx[2][1], self.mtx[2][2] + other.mtx[2][2], self.mtx[2][3] + other.mtx[2][3]],
            [self.mtx[3][0] + other.mtx[3][0], self.mtx[3][1] + other.mtx[3][1], self.mtx[3][2] + other.mtx[3][2], self.mtx[3][3] + other.mtx[3][3]]
        )

    def __sub__(self, other):
        if not isinstance(other, Matrix4):
            return NotImplemented
        return Matrix4(
            [self.mtx[0][0] - other.mtx[0][0], self.mtx[0][1] - other.mtx[0][1], self.mtx[0][2] - other.mtx[0][2], self.mtx[0][3] - other.mtx[0][3]],
            [self.mtx[1][0] - other.mtx[1][0], self.mtx[1][1] - other.mtx[1][1], self.mtx[1][2] - other.mtx[1][2], self.mtx[1][3] - other.mtx[1][3]],
            [self.mtx[2][0] - other.mtx[2][0], self.mtx[2][1] - other.mtx[2][1], self.mtx[2][2] - other.mtx[2][2], self.mtx[2][3] - other.mtx[2][3]],
            [self.mtx[3][0] - other.mtx[3][0], self.mtx[3][1] - other.mtx[3][1], self.mtx[3][2] - other.mtx[3][2], self.mtx[3][3] - other.mtx[3][3]]
        )

    def __truediv__(self, other):
        if not isinstance(other, Real):
            return NotImplemented
        return Matrix4(
            [self.mtx[0][0] / other, self.mtx[0][1] / other, self.mtx[0][2] / other, self.mtx[0][3] / other],
            [self.mtx[1][0] / other, self.mtx[1][1] / other, self.mtx[1][2] / other, self.mtx[1][3] / other],
            [self.mtx[2][0] / other, self.mtx[2][1] / other, self.mtx[2][2] / other, self.mtx[2][3] / other],
            [self.mtx[3][0] / other, self.mtx[3][1] / other, self.mtx[3][2] / other, self.mtx[3][3] / other]
        )

    def determinant(self):
        """Calculate the determinant of the matrix."""
        return (self.mtx[0][0] * self.mtx[1][1] - self.mtx[1][0] * self.mtx[0][1]) \
             * (self.mtx[2][2] * self.mtx[3][3] - self.mtx[3][2] * self.mtx[2][3]) \
             - (self.mtx[0][0] * self.mtx[2][1] - self.mtx[2][0] * self.mtx[0][1]) \
             * (self.mtx[1][2] * self.mtx[3][3] - self.mtx[3][2] * self.mtx[1][3]) \
             + (self.mtx[0][0] * self.mtx[3][1] - self.mtx[3][0] * self.mtx[0][1]) \
             * (self.mtx[1][2] * self.mtx[2][3] - self.mtx[2][2] * self.mtx[1][3]) \
             + (self.mtx[1][0] * self.mtx[2][1] - self.mtx[2][0] * self.mtx[1][1]) \
             * (self.mtx[0][2] * self.mtx[3][3] - self.mtx[3][2] * self.mtx[0][3]) \
             - (self.mtx[1][0] * self.mtx[3][1] - self.mtx[3][0] * self.mtx[1][1]) \
             * (self.mtx[0][2] * self.mtx[2][3] - self.mtx[2][2] * self.mtx[0][3]) \
             + (self.mtx[2][0] * self.mtx[3][1] - self.mtx[3][0] * self.mtx[2][1]) \
             * (self.mtx[0][2] * self.mtx[1][3] - self.mtx[1][2] * self.mtx[0][3])

    def from_axes(self, x_axis, y_axis, z_axis):
        """Create a rotation matrix from 3 basis vectors (x_axis, y_axis, and z_axis)."""
        if not isinstance(x, Vector4) or not isinstance(y, Vector4) or not isinstance(z, Vector4):
            raise NotImplementedError
        self.mtx[0][0] = x_axis.x
        self.mtx[0][1] = x_axis.y
        self.mtx[0][2] = x_axis.z
        self.mtx[0][3] = 0.0
        self.mtx[1][0] = y_axis.x
        self.mtx[1][1] = y_axis.y
        self.mtx[1][2] = y_axis.z
        self.mtx[1][3] = 0.0
        self.mtx[2][0] = z_axis.x
        self.mtx[2][1] = z_axis.y
        self.mtx[2][2] = z_axis.z
        self.mtx[2][3] = 0.0
        self.mtx[3][0] = 0.0
        self.mtx[3][1] = 0.0
        self.mtx[3][2] = 0.0
        self.mtx[3][3] = 1.0

    def from_axes_transposed(self, x_axis, y_axis, z_axis):
        """Create a transposed rotation matrix from 3 basis vectors (x_axis, y_axis, and z_axis)."""
        if not isinstance(x, Vector4) or not isinstance(y, Vector4) or not isinstance(z, Vector4):
            raise NotImplementedError
        self.mtx[0][0] = x_axis.x
        self.mtx[0][1] = y_axis.x
        self.mtx[0][2] = z_axis.x
        self.mtx[0][3] = 0.0
        self.mtx[1][0] = x_axis.y
        self.mtx[1][1] = y_axis.y
        self.mtx[1][2] = z_axis.y
        self.mtx[1][3] = 0.0
        self.mtx[2][0] = x_axis.z
        self.mtx[2][1] = y_axis.z
        self.mtx[2][2] = z_axis.z
        self.mtx[2][3] = 0.0
        self.mtx[3][0] = 0.0
        self.mtx[3][1] = 0.0
        self.mtx[3][2] = 0.0
        self.mtx[3][3] = 1.0

    def from_heading_pitch_roll(self, heading_degrees, pitch_degrees, roll_degrees):
        """
        Create a rotation matrix based on a Euler Transform.
        The popular NASA standard airplane convention of heading-pitch-roll (i.e., RzRxRy) is used here.
        """
        if not isinstance(heading_degrees, Real) or not isinstance(pitch_degrees, Real) or not isinstance(roll_degrees, Real):
            raise NotImplementedError

        heading = math.radians(heading_degrees)
        pitch = math.radians(pitch_degrees)
        roll = math.radians(roll_degrees)

        cos_heading = math.cos(heading)
        cos_pitch = math.cos(pitch)
        cos_roll = math.cos(roll)
        sin_heading = math.sin(heading)
        sin_pitch = math.sin(pitch)
        sin_roll = math.sin(roll)

        self.mtx[0][0] = cos_roll * cos_heading - sin_roll * sin_pitch * sin_heading
        self.mtx[0][1] = sin_roll * cos_heading + cos_roll * sin_pitch * sin_heading
        self.mtx[0][2] = -cos_pitch * sin_heading
        self.mtx[0][3] = 0.0
        self.mtx[1][0] = -sin_roll * cos_pitch
        self.mtx[1][1] = cos_roll * cos_pitch
        self.mtx[1][2] = sin_pitch
        self.mtx[1][3] = 0.0
        self.mtx[2][0] = cos_roll * sin_heading + sin_roll * sin_pitch * cos_heading
        self.mtx[2][1] = sin_roll * sin_heading - cos_roll * sin_pitch * cos_heading
        self.mtx[2][2] = cos_pitch * cos_heading
        self.mtx[2][3] = 0.0
        self.mtx[3][0] = 0.0
        self.mtx[3][1] = 0.0
        self.mtx[3][2] = 0.0
        self.mtx[3][3] = 1.0

    def inverse(self):
        """
        Calculate the inverse of the matrix. If the inverse doesn't exist, the identity matrix is returned instead.
        This method of computing the inverse of a 4x4 matrix is based on a similar function found in Paul Nettle's matrix template class.
        """
        result = Matrix4()
        d = self.determinant()

        if close_enough(d, 0.0):
            result.mtx[0][0] = 1.0
            result.mtx[0][1] = 0.0
            result.mtx[0][2] = 0.0
            result.mtx[0][3] = 0.0
            result.mtx[1][0] = 0.0
            result.mtx[1][1] = 1.0
            result.mtx[1][2] = 0.0
            result.mtx[1][3] = 0.0
            result.mtx[2][0] = 0.0
            result.mtx[2][1] = 0.0
            result.mtx[2][2] = 1.0
            result.mtx[2][3] = 0.0
            result.mtx[3][0] = 0.0
            result.mtx[3][1] = 0.0
            result.mtx[3][2] = 0.0
            result.mtx[3][3] = 1.0
        else:
            d = 1.0 / d;

            result.mtx[0][0] = d * (self.mtx[1][1] * (self.mtx[2][2] * self.mtx[3][3] - self.mtx[3][2] * self.mtx[2][3]) + self.mtx[2][1] * (self.mtx[3][2] * self.mtx[1][3] - self.mtx[1][2] * self.mtx[3][3]) + self.mtx[3][1] * (self.mtx[1][2] * self.mtx[2][3] - self.mtx[2][2] * self.mtx[1][3]))
            result.mtx[1][0] = d * (self.mtx[1][2] * (self.mtx[2][0] * self.mtx[3][3] - self.mtx[3][0] * self.mtx[2][3]) + self.mtx[2][2] * (self.mtx[3][0] * self.mtx[1][3] - self.mtx[1][0] * self.mtx[3][3]) + self.mtx[3][2] * (self.mtx[1][0] * self.mtx[2][3] - self.mtx[2][0] * self.mtx[1][3]))
            result.mtx[2][0] = d * (self.mtx[1][3] * (self.mtx[2][0] * self.mtx[3][1] - self.mtx[3][0] * self.mtx[2][1]) + self.mtx[2][3] * (self.mtx[3][0] * self.mtx[1][1] - self.mtx[1][0] * self.mtx[3][1]) + self.mtx[3][3] * (self.mtx[1][0] * self.mtx[2][1] - self.mtx[2][0] * self.mtx[1][1]))
            result.mtx[3][0] = d * (self.mtx[1][0] * (self.mtx[3][1] * self.mtx[2][2] - self.mtx[2][1] * self.mtx[3][2]) + self.mtx[2][0] * (self.mtx[1][1] * self.mtx[3][2] - self.mtx[3][1] * self.mtx[1][2]) + self.mtx[3][0] * (self.mtx[2][1] * self.mtx[1][2] - self.mtx[1][1] * self.mtx[2][2]))

            result.mtx[0][1] = d * (self.mtx[2][1] * (self.mtx[0][2] * self.mtx[3][3] - self.mtx[3][2] * self.mtx[0][3]) + self.mtx[3][1] * (self.mtx[2][2] * self.mtx[0][3] - self.mtx[0][2] * self.mtx[2][3]) + self.mtx[0][1] * (self.mtx[3][2] * self.mtx[2][3] - self.mtx[2][2] * self.mtx[3][3]))
            result.mtx[1][1] = d * (self.mtx[2][2] * (self.mtx[0][0] * self.mtx[3][3] - self.mtx[3][0] * self.mtx[0][3]) + self.mtx[3][2] * (self.mtx[2][0] * self.mtx[0][3] - self.mtx[0][0] * self.mtx[2][3]) + self.mtx[0][2] * (self.mtx[3][0] * self.mtx[2][3] - self.mtx[2][0] * self.mtx[3][3]))
            result.mtx[2][1] = d * (self.mtx[2][3] * (self.mtx[0][0] * self.mtx[3][1] - self.mtx[3][0] * self.mtx[0][1]) + self.mtx[3][3] * (self.mtx[2][0] * self.mtx[0][1] - self.mtx[0][0] * self.mtx[2][1]) + self.mtx[0][3] * (self.mtx[3][0] * self.mtx[2][1] - self.mtx[2][0] * self.mtx[3][1]))
            result.mtx[3][1] = d * (self.mtx[2][0] * (self.mtx[3][1] * self.mtx[0][2] - self.mtx[0][1] * self.mtx[3][2]) + self.mtx[3][0] * (self.mtx[0][1] * self.mtx[2][2] - self.mtx[2][1] * self.mtx[0][2]) + self.mtx[0][0] * (self.mtx[2][1] * self.mtx[3][2] - self.mtx[3][1] * self.mtx[2][2]))

            result.mtx[0][2] = d * (self.mtx[3][1] * (self.mtx[0][2] * self.mtx[1][3] - self.mtx[1][2] * self.mtx[0][3]) + self.mtx[0][1] * (self.mtx[1][2] * self.mtx[3][3] - self.mtx[3][2] * self.mtx[1][3]) + self.mtx[1][1] * (self.mtx[3][2] * self.mtx[0][3] - self.mtx[0][2] * self.mtx[3][3]))
            result.mtx[1][2] = d * (self.mtx[3][2] * (self.mtx[0][0] * self.mtx[1][3] - self.mtx[1][0] * self.mtx[0][3]) + self.mtx[0][2] * (self.mtx[1][0] * self.mtx[3][3] - self.mtx[3][0] * self.mtx[1][3]) + self.mtx[1][2] * (self.mtx[3][0] * self.mtx[0][3] - self.mtx[0][0] * self.mtx[3][3]))
            result.mtx[2][2] = d * (self.mtx[3][3] * (self.mtx[0][0] * self.mtx[1][1] - self.mtx[1][0] * self.mtx[0][1]) + self.mtx[0][3] * (self.mtx[1][0] * self.mtx[3][1] - self.mtx[3][0] * self.mtx[1][1]) + self.mtx[1][3] * (self.mtx[3][0] * self.mtx[0][1] - self.mtx[0][0] * self.mtx[3][1]))
            result.mtx[3][2] = d * (self.mtx[3][0] * (self.mtx[1][1] * self.mtx[0][2] - self.mtx[0][1] * self.mtx[1][2]) + self.mtx[0][0] * (self.mtx[3][1] * self.mtx[1][2] - self.mtx[1][1] * self.mtx[3][2]) + self.mtx[1][0] * (self.mtx[0][1] * self.mtx[3][2] - self.mtx[3][1] * self.mtx[0][2]))

            result.mtx[0][3] = d * (self.mtx[0][1] * (self.mtx[2][2] * self.mtx[1][3] - self.mtx[1][2] * self.mtx[2][3]) + self.mtx[1][1] * (self.mtx[0][2] * self.mtx[2][3] - self.mtx[2][2] * self.mtx[0][3]) + self.mtx[2][1] * (self.mtx[1][2] * self.mtx[0][3] - self.mtx[0][2] * self.mtx[1][3]))
            result.mtx[1][3] = d * (self.mtx[0][2] * (self.mtx[2][0] * self.mtx[1][3] - self.mtx[1][0] * self.mtx[2][3]) + self.mtx[1][2] * (self.mtx[0][0] * self.mtx[2][3] - self.mtx[2][0] * self.mtx[0][3]) + self.mtx[2][2] * (self.mtx[1][0] * self.mtx[0][3] - self.mtx[0][0] * self.mtx[1][3]))
            result.mtx[2][3] = d * (self.mtx[0][3] * (self.mtx[2][0] * self.mtx[1][1] - self.mtx[1][0] * self.mtx[2][1]) + self.mtx[1][3] * (self.mtx[0][0] * self.mtx[2][1] - self.mtx[2][0] * self.mtx[0][1]) + self.mtx[2][3] * (self.mtx[1][0] * self.mtx[0][1] - self.mtx[0][0] * self.mtx[1][1]))
            result.mtx[3][3] = d * (self.mtx[0][0] * (self.mtx[1][1] * self.mtx[2][2] - self.mtx[2][1] * self.mtx[1][2]) + self.mtx[1][0] * (self.mtx[2][1] * self.mtx[0][2] - self.mtx[0][1] * self.mtx[2][2]) + self.mtx[2][0] * (self.mtx[0][1] * self.mtx[1][2] - self.mtx[1][1] * self.mtx[0][2]))

        return result

    def rotate(self, axis, degrees):
        """
        Create a rotation matrix about the specified axis.
        The axis must be a unit vector and the angle must be in degrees.

        Let u = axis of rotation = (x, y, z)

                    | x^2(1 - c) + c  xy(1 - c) + zs  xz(1 - c) - ys   0 |
        Ru(angle) = | yx(1 - c) - zs  y^2(1 - c) + c  yz(1 - c) + xs   0 |
                    | zx(1 - c) - ys  zy(1 - c) - xs  z^2(1 - c) + c   0 |
                    |      0              0                0           1 |

        where
            c = cos(angle)
            s = sin(angle)
        """
        if not isinstance(axis, Vector4) or not isinstance(degrees, Real):
            raise NotImplementedError

        angle = math.radians(degrees);

        x = axis.x
        y = axis.y
        z = axis.z
        c = math.cos(angle)
        s = math.sin(angle)

        self.mtx[0][0] = (x * x) * (1.0 - c) + c
        self.mtx[0][1] = (x * y) * (1.0 - c) + (z * s)
        self.mtx[0][2] = (x * z) * (1.0 - c) - (y * s)
        self.mtx[0][3] = 0.0
        self.mtx[1][0] = (y * x) * (1.0 - c) - (z * s)
        self.mtx[1][1] = (y * y) * (1.0 - c) + c
        self.mtx[1][2] = (y * z) * (1.0 - c) + (x * s)
        self.mtx[1][3] = 0.0
        self.mtx[2][0] = (z * x) * (1.0 - c) + (y * s)
        self.mtx[2][1] = (z * y) * (1.0 - c) - (x * s)
        self.mtx[2][2] = (z * z) * (1.0 - c) + c
        self.mtx[2][3] = 0.0
        self.mtx[3][0] = 0.0
        self.mtx[3][1] = 0.0
        self.mtx[3][2] = 0.0
        self.mtx[3][3] = 1.0

    def scale(self, sx, sy, sz):
        """
        Create a scaling matrix.

                        | sx   0    0    0 |
        S(sx, sy, sz) = | 0    sy   0    0 |
                        | 0    0    sz   0 |
                        | 0    0    0    1 |
        """
        self.mtx[0][0] = sx
        self.mtx[0][1] = 0.0
        self.mtx[0][2] = 0.0
        self.mtx[0][3] = 0.0
        self.mtx[1][0] = 0.0
        self.mtx[1][1] = sy
        self.mtx[1][2] = 0.0
        self.mtx[1][3] = 0.0
        self.mtx[2][0] = 0.0
        self.mtx[2][1] = 0.0
        self.mtx[2][2] = sz
        self.mtx[2][3] = 0.0
        self.mtx[3][0] = 0.0
        self.mtx[3][1] = 0.0
        self.mtx[3][2] = 0.0
        self.mtx[3][3] = 1.0

    def to_axes(self):
        """
        Extract the local x, y, and z axes from the matrix.
        The x, y, and z axes are returned in a 3-element tuple of Vector4 objects in that order.
        This only makes sense for rotation matrices.
        Calling this method on a matrix that isn't a rotation matrix will lead to undefined behavior.
        """
        x_axis = Vector4(self.mtx[0][0], self.mtx[0][1], self.mtx[0][2], 0.0)
        y_axis = Vector4(self.mtx[1][0], self.mtx[1][1], self.mtx[1][2], 0.0)
        z_axis = Vector4(self.mtx[2][0], self.mtx[2][1], self.mtx[2][2], 0.0)
        return (x_axis, y_axis, z_axis)

    def to_axes_transposed(self):
        """
        Extract the local x, y, and z axes from the transpose of the matrix.
        The x, y, and z axes are returned in a 3-element tuple of Vector4 objects in that order.
        This only makes sense for rotation matrices.
        Calling this method on a matrix that isn't a rotation matrix will lead to undefined behavior.
        """
        x_axis = Vector4(self.mtx[0][0], self.mtx[1][0], self.mtx[2][0], 0.0)
        y_axis = Vector4(self.mtx[0][1], self.mtx[1][1], self.mtx[2][1], 0.0)
        z_axis = Vector4(self.mtx[0][2], self.mtx[1][2], self.mtx[2][2], 0.0)
        return (x_axis, y_axis, z_axis)

    def to_heading_pitch_roll(self):
        """
        Extract the Euler angles from a rotation matrix.
        The returned angles are in degrees.
        This method might suffer from numerical imprecision for ill defined rotation matrices.
        This function only works for rotation matrices constructed using the popular NASA standard airplane convention of heading-pitch-roll (i.e., RzRxRy).

        The algorithm used is from:
            David Eberly, "Euler Angle Formulas", Geometric Tools web site, http://www.geometrictools.com/Documentation/EulerAngles.pdf.

        The heading, pitch, and roll angles are returned in a 3-element tuple in that order.
        """
        theta_x = math.asin(self.mtx[1][2])
        theta_y = 0.0
        theta_z = 0.0
        half_pi = math.pi * 0.5

        if theta_x < half_pi:
            if theta_x > -half_pi:
                theta_z = math.atan2(-self.mtx[1][0], self.mtx[1][1])
                theta_y = math.atan2(-self.mtx[0][2], self.mtx[2][2])
            else:
                # Not a unique solution.
                theta_z = -math.atan2(self.mtx[2][0], self.mtx[0][0])
                theta_y = 0.0
        else:
            # Not a unique solution.
            theta_z = math.atan2(self.mtx[2][0], self.mtx[0][0])
            theta_y = 0.0

        heading = math.degrees(theta_y)
        pitch = math.degrees(theta_x)
        roll = math.degrees(theta_z)
        return (heading, pitch, roll)

    def translate(self, tx, ty, tz):
        """
        Create a translation matrix.

                        | 1    0    0    0 |
        T(tx, ty, tz) = | 0    1    0    0 |
                        | 0    0    1    0 |
                        | tx   ty   tz   1 |
        """
        self.mtx[0][0] = 1.0
        self.mtx[0][1] = 0.0
        self.mtx[0][2] = 0.0
        self.mtx[0][3] = 0.0
        self.mtx[1][0] = 0.0
        self.mtx[1][1] = 1.0
        self.mtx[1][2] = 0.0
        self.mtx[1][3] = 0.0
        self.mtx[2][0] = 0.0
        self.mtx[2][1] = 0.0
        self.mtx[2][2] = 1.0
        self.mtx[2][3] = 0.0
        self.mtx[3][0] = tx
        self.mtx[3][1] = ty
        self.mtx[3][2] = tz
        self.mtx[3][3] = 1.0

    def transpose(self):
        """Return the transpose of the matrix."""
        return Matrix4(
            [self.mtx[0][0], self.mtx[1][0], self.mtx[2][0], self.mtx[3][0]],
            [self.mtx[0][1], self.mtx[1][1], self.mtx[2][1], self.mtx[3][1]],
            [self.mtx[0][2], self.mtx[1][2], self.mtx[2][2], self.mtx[3][2]],
            [self.mtx[0][3], self.mtx[1][3], self.mtx[2][3], self.mtx[3][3]]
        )

    @staticmethod
    def create_from_axes(x_axis, y_axis, z_axis):
        """Create a rotation matrix using the 3 provided basis vectors."""
        result = Matrix4()
        result.from_axes(x_axis, y_axis, z_axis)
        return result

    @staticmethod
    def create_from_axes_transposed(x_axis, y_axis, z_axis):
        """Create a transposed rotation matrix using the 3 provided basis vectors."""
        result = Matrix4()
        result.from_axes_transposed(x_axis, y_axis, z_axis)
        return result

    @staticmethod
    def create_from_heading_pitch_roll(heading_degrees, pitch_degrees, roll_degrees):
        """Create a rotation matrix based on a Euler Transform."""
        result = Matrix4()
        result.from_heading_pitch_roll(heading_degrees, pitch_degrees, roll_degrees)
        return result

    @staticmethod
    def create_mirror(plane_normal, point_on_plane):
        """
        Construct a reflection (or mirror) matrix given an arbitrary plane that passes through the specified position.

        The algorithm used is from:
            Ronald Goldman, "Matrices and Transformation," Graphics Gems, 1990.
        """
        if not isinstance(plane_normal, Vector4) or not isinstance(point_on_plane, Vector4):
            raise NotImplementedError

        x = plane_normal.x
        y = plane_normal.y
        z = plane_normal.z
        dot = Vector4.dot(plane_normal, point_on_plane)

        return Matrix4(
            [(1.0 - 2.0 * x * x), (-2.0 * y * x), (-2.0 * z * x), 0.0],
            [(-2.0 * x * y), (1.0 - 2.0 * y * y), (-2.0 * z * y), 0.0],
            [(-2.0 * x * z), (-2.0 * y * z), (1.0 - 2.0 * z * z), 0.0],
            [(2.0 * dot * x), (2.0 * dot * y), (2.0 * dot * z), 1.0]
        )

    @staticmethod
    def create_rotate(axis, degrees):
        """Create a rotation matrix about the specified axis. The axis must be a unit vector and the angle must be in degrees."""
        result = Matrix4()
        result.rotate(axis, degrees)
        return result

    @staticmethod
    def create_scale(sx, sy, sz):
        """Create a scaling matrix."""
        result = Matrix4()
        result.scale(sx, sy, sz)
        return result

    @staticmethod
    def create_translate(tx, ty, tz):
        """Create a translation matrix."""
        result = Matrix4()
        result.translate(tx, ty, tz)
        return result

    @staticmethod
    def identity():
        """Create an identity matrix."""
        return Matrix4([1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0])


class Quaternion(object):
    """
    A quaternion has a scalar component (w) and a 3D vector component (x, y, z).
    Quaternion are multiplied in a left to right order.
    """
    def __init__(self, w=0.0, x=0.0, y=0.0, z=0.0):
        self.w = w
        self.x = x
        self.y = y
        self.z = z

    def __str__(self):
        return f"({self.w}, {self.x}, {self.y}, {self.z})"

    def __eq__(self, other):
        if not isinstance(other, Quaternion):
            return NotImplemented
        return close_enough(self.w, other.w) and close_enough(self.x, other.x) and close_enough(self.y, other.y) and close_enough(self.z, other.z)

    def __ne__(self, other):
        if not isinstance(other, Quaternion):
            return NotImplemented
        return not self == other

    def __mul__(self, other):
        if isinstance(other, Real):
            return Quaternion(self.w * other, self.x * other, self.y * other, self.z * other)
        elif isinstance(other, Quaternion):
            return Quaternion(
                (self.w * other.w) - (self.x * other.x) - (self.y * other.y) - (self.z * other.z),
                (self.w * other.x) + (self.x * other.w) - (self.y * other.z) + (self.z * other.y),
                (self.w * other.y) + (self.x * other.z) + (self.y * other.w) - (self.z * other.x),
                (self.w * other.z) - (self.x * other.y) + (self.y * other.x) + (self.z * other.w)
            )
        else:
            return NotImplemented

    def __rmul__(self, other):
        return self * other

    def __add__(self, other):
        if not isinstance(other, Quaternion):
            return NotImplemented
        return Quaternion(self.w + other.w, self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other):
        if not isinstance(other, Quaternion):
            return NotImplemented
        return Quaternion(self.w - other.w, self.x - other.x, self.y - other.y, self.z - other.z)

    def __truediv__(self, other):
        if not isinstance(other, Real):
            return NotImplemented
        return Quaternion(self.w / other, self.x / other, self.y / other, self.z / other)

    def conjugate(self):
        """
        Return the conjugate of the quaternion.
        The conjugate is obtained by negating the vector portion of the quaternion.
        """
        return Quaternion(self.w, -self.x, -self.y, -self.z)

    def from_axis_angle(self, axis, degrees):
        """Create a quaternion from an axis of rotation (Vector3) and an angle (in degrees)."""
        if not isinstance(axis, Vector3) or not isinstance(degrees, Real):
            raise NotImplementedError

        half_theta = math.radians(degrees) * 0.5
        s = math.sin(half_theta)

        self.w = math.cos(half_theta)
        self.x = axis.x * s
        self.y = axis.y * s
        self.z = axis.z * s

    def from_matrix(self, matrix):
        """
        Creates a quaternion from a rotation matrix (Matrix3, or Matrix4).

        The algorithm used is from:
            Allan and Mark Watt's "Advanced Animation and Rendering Techniques" (ACM Press 1992).
        """
        if not isinstance(matrix, Matrix3) and not isinstance(matrix, Matrix4):
            raise NotImplementedError

        s = 0.0
        q = [0.0, 0.0, 0.0, 0.0]
        trace = matrix.mtx[0][0] + matrix.mtx[1][1] + matrix.mtx[2][2]

        if trace > 0.0:
            s = math.sqrt(trace + 1.0)
            q[3] = s * 0.5
            s = 0.5 / s
            q[0] = (matrix.mtx[1][2] - matrix.mtx[2][1]) * s
            q[1] = (matrix.mtx[2][0] - matrix.mtx[0][2]) * s
            q[2] = (matrix.mtx[0][1] - matrix.mtx[1][0]) * s
        else:
            nxt = [1, 2, 0]
            i = 0
            j = 0
            k = 0

            if matrix.mtx[1][1] > matrix.mtx[0][0]:
                i = 1

            if matrix.mtx[2][2] > matrix.mtx[i][i]:
                i = 2

            j = nxt[i]
            k = nxt[j]
            s = math.sqrt((matrix.mtx[i][i] - (matrix.mtx[j][j] + matrix.mtx[k][k])) + 1.0)

            q[i] = s * 0.5
            s = 0.5 / s
            q[3] = (matrix.mtx[j][k] - matrix.mtx[k][j]) * s
            q[j] = (matrix.mtx[i][j] + matrix.mtx[j][i]) * s
            q[k] = (matrix.mtx[i][k] + matrix.mtx[k][i]) * s

        self.x = q[0]
        self.y = q[1]
        self.z = q[2]
        self.w = q[3]

    def from_heading_pitch_roll(self, heading_degrees, pitch_degrees, roll_degrees):
        """
        Create a rotation for the quaternion based on a Euler Transform.
        The popular NASA standard airplane convention of heading-pitch-roll (i.e., RzRxRy) is used here.
        """
        matrix = Matrix3.create_from_heading_pitch_roll(heading_degrees, pitch_degrees, roll_degrees)
        self.from_matrix(matrix)

    def magnitude(self):
        """Return the magnitude of the quaternion."""
        return math.sqrt(self.w * self.w + self.x * self.x + self.y * self.y + self.z * self.z)

    def inverse(self):
        """Return the inverse of the quaternion."""
        inv_mag = 1.0 / self.magnitude()
        return self.conjugate() * inv_mag

    def normalize(self):
        """Normalize the quaternion so that it can be used to represent rotations."""
        inv_mag = 1.0 / self.magnitude()
        self.w *= inv_mag
        self.x *= inv_mag
        self.y *= inv_mag
        self.z *= inv_mag

    def to_axis_angle(self):
        """
        Convert the quaternion to an axis and angle.
        The axis and angle is returned in a 2-element tuple in that order.
        The returned axis is a Vector3 object and the angle is a float in degrees.
        """
        axis = Vector3()
        degrees = 0.0
        sin_half_theta_sq = 1.0 - self.w * self.w

        # Guard against numerical imprecision and identity quaternions.
        if sin_half_theta_sq <= 0.0:
            axis.x = 1.0
            axis.y = 0.0
            axis.z = 0.0
            degrees = 0.0
        else:
            inv_sin_half_theta = 1.0 / math.sqrt(sin_half_theta_sq)

            axis.x = self.x * inv_sin_half_theta
            axis.y = self.y * inv_sin_half_theta
            axis.z = self.z * inv_sin_half_theta
            degrees = math.degrees(2.0 * math.acos(self.w))

        return (axis, degrees)

    def to_matrix3(self):
        """
        Convert the quaternion to a rotation matrix.
            | 1 - 2(y^2 + z^2)	2(xy + wz)			2(xz - wy)		 |
            | 2(xy - wz)		1 - 2(x^2 + z^2)	2(yz + wx)		 |
            | 2(xz + wy)		2(yz - wx)			1 - 2(x^2 + y^2) |
        """
        x2 = self.x + self.x
        y2 = self.y + self.y
        z2 = self.z + self.z
        xx = self.x * x2
        xy = self.x * y2
        xz = self.x * z2
        yy = self.y * y2
        yz = self.y * z2
        zz = self.z * z2
        wx = self.w * x2
        wy = self.w * y2
        wz = self.w * z2

        return Matrix3(
            [(1.0 - (yy + zz)), (xy + wz), (xz - wy)],
            [(xy - wz), (1.0 - (xx + zz)), (yz + wx)],
            [(xz + wy), (yz - wx), (1.0 - (xx + yy))]
        )

    def to_matrix4(self):
        """
        Converts this quaternion to a rotation matrix.
            | 1 - 2(y^2 + z^2)	2(xy + wz)			2(xz - wy)			0  |
            | 2(xy - wz)		1 - 2(x^2 + z^2)	2(yz + wx)			0  |
            | 2(xz + wy)		2(yz - wx)			1 - 2(x^2 + y^2)	0  |
            | 0					0					0					1  |
        """
        x2 = self.x + self.x
        y2 = self.y + self.y
        z2 = self.z + self.z
        xx = self.x * x2
        xy = self.x * y2
        xz = self.x * z2
        yy = self.y * y2
        yz = self.y * z2
        zz = self.z * z2
        wx = self.w * x2
        wy = self.w * y2
        wz = self.w * z2

        return Matrix4(
            [(1.0 - (yy + zz)), (xy + wz), (xz - wy), 0.0],
            [(xy - wz), (1.0 - (xx + zz)), (yz + wx), 0.0],
            [(xz + wy), (yz - wx), (1.0 - (xx + yy)), 0.0],
            [0.0, 0.0, 0.0, 1.0]
        )

    def to_heading_pitch_roll(self):
        """
        Extract the Euler angles from the quaternion.
        The returned angles are in degrees.
        The heading, pitch, and roll angles are returned in a 3-element tuple in that order.
        """
        m = self.to_matrix3()
        return m.to_heading_pitch_roll()

    @staticmethod
    def create_from_axis_angle(axis, degrees):
        """Create a quaternion that represents a rotation about an axis."""
        result = Quaternion()
        result.from_axis_angle(axis, degrees)
        return result

    @staticmethod
    def create_from_heading_pitch_roll(heading_degrees, pitch_degrees, roll_degrees):
        """Create a quaternion based on a Euler Transform."""
        result = Quaternion()
        result.from_heading_pitch_roll(heading_degrees, pitch_degrees, roll_degrees)
        return result

    @staticmethod
    def create_from_matrix(matrix):
        """Create a quaternion from a Matrix3 or Matrix4 object."""
        result = Quaternion()
        result.from_matrix(matrix)
        return result

    @staticmethod
    def identity():
        """Create an identity quaternion."""
        return Quaternion(1.0, 0.0, 0.0, 0.0)

    @staticmethod
    def slerp(a, b, t):
        """
        Smoothly interpolates from quaternion 'a' to quaternion 'b' using spherical linear interpolation as t varies from 0 to 1.

        Both quaternions must be unit length and represent absolute rotations.
        In particular quaternion 'b' must not be relative to quaternion 'a'.
        If 'b' is relative to 'a' make 'b' an absolute rotation by: b = a * b.

        The interpolation parameter 't' is in the range [0,1].
        When t = 0 the resulting quaternion will be 'a'.
        When t = 1 the resulting quaternion will be 'b'.

        The algorithm used is adapted from:
            Allan and Mark Watt's "Advanced Animation and Rendering Techniques" (ACM Press 1992).
        """
        if not isinstance(a, Quaternion) or not isinstance(b, Quaternion) or not isinstance(t, Real):
            raise NotImplementedError

        result = Quaternion()
        omega = 0.0
        cosom = (a.x * b.x) + (a.y * b.y) + (a.z * b.z) + (a.w * b.w)
        sinom = 0.0
        scale0 = 0.0
        scale1 = 0.0
        epsilon = 1e-6
        half_pi = math.pi * 0.5

        if (1.0 + cosom) > epsilon:
            # 'a' and 'b' quaternions are not opposite each other.
            if (1.0 - cosom) > epsilon:
                # Standard case - slerp.
                omega = math.acos(cosom)
                sinom = math.sin(omega)
                scale0 = math.sin((1.0 - t) * omega) / sinom
                scale1 = math.sin(t * omega) / sinom
            else:
                # 'a' and 'b' quaternions are very close so lerp instead.
                scale0 = 1.0 - t
                scale1 = t

            result.x = scale0 * a.x + scale1 * b.x
            result.y = scale0 * a.y + scale1 * b.y
            result.z = scale0 * a.z + scale1 * b.z
            result.w = scale0 * a.w + scale1 * b.w
        else:
            # 'a' and 'b' quaternions are opposite each other.
            result.x = -b.y
            result.y = b.x
            result.z = -b.w
            result.w = b.z

            scale0 = math.sin((1.0 - t) - half_pi)
            scale1 = math.sin(t * half_pi)

            result.x = scale0 * a.x + scale1 * result.x
            result.y = scale0 * a.y + scale1 * result.y
            result.z = scale0 * a.z + scale1 * result.z
            result.w = scale0 * a.w + scale1 * result.w

        return result


class BoundingBox(object):
    """
    A bounding box class used for collision detection.
    """
    def __init__(self, min=Vector3(0.0, 0.0, 0.0), max=Vector3(0.0, 0.0, 0.0)):
        if not isinstance(min, Vector3) or not isinstance(max, Vector3):
            raise NotImplementedError

        self.min = Vector3(min.x, min.y, min.z)
        self.max = Vector3(max.x, max.y, max.z)

    def __str__(self):
        return f"[min={self.min}, max={self.max}]"

    def center(self):
        """Calculate the center position of the bounding box."""
        return (self.min + self.max) * 0.5

    def size(self):
        """Calculate the length of the longest diagonal of the bounding box."""
        return (self.max - self.min).magnitude()

    def radius(self):
        """The radius of the bounding box is also the radius of the bounding sphere around the bounding box."""
        return self.size() * 0.5

    def has_collided(self, other):
        """
        Determines whether these two bounding boxes have collided with each other.
        AABB (Axis-Aligned Bounding Box) collision detection is used here.
        """
        if not isinstance(other, BoundingBox):
            raise NotImplementedError

        return  self.min.x <= other.max.x \
            and self.max.x >= other.min.x \
            and self.min.y <= other.max.y \
            and self.max.y >= other.min.y \
            and self.min.z <= other.max.z \
            and self.max.z >= other.min.z

    def point_in(self, point):
        """Determines whether the 3D point is in the bounding box."""
        if not isinstance(point, Vector3) and not isinstance(point, Vector4):
            raise NotImplementedError

        return  point.x >= self.min.x \
            and point.x <= self.max.x \
            and point.y >= self.min.y \
            and point.y <= self.max.y \
            and point.z >= self.min.z \
            and point.z <= self.max.z


class BoundingSphere(object):
    """
    A bounding sphere class used for collision detection.
    """
    def __init__(self, center=Vector3(0.0, 0.0, 0.0), radius=0.0):
        if not isinstance(center, Vector3) or not isinstance(radius, Real):
            raise NotImplementedError

        self.center = Vector3(center.x, center.y, center.z)
        self.radius = radius

    def __str__(self):
        return f"[center={self.center}, radius={self.radius}]"

    def has_collided(self, other):
        """Determine whether these two bounding spheres have collide with each other."""
        if not isinstance(other, BoundingSphere):
            raise NotImplementedError

        distance = other.center - self.center
        length_sq = (distance.x * distance.x) + (distance.y * distance.y) + (distance.z * distance.z)
        radii_sq = (other.radius + self.radius) * (other.radius + self.radius)

        return length_sq < radii_sq

    def point_in(self, point):
        """Determines whether the 3D point is in the bounding sphere."""
        if not isinstance(point, Vector3):
            raise NotImplementedError

        return Vector3.distancesq(point, self.center) < (self.radius * self.radius)


class Plane(object):
    """
    A plane class used for collision detection.
    """
    def __init__(self, a=0.0, b=0.0, c=0.0, d=0.0):
        """
        The plane equation is: ax + by + cz + d = 0
        'normal' is the plane's normal (a, b, c).
        'd' is the distance of the plane along its normal from the origin.
        """
        if not isinstance(a, Real) or not isinstance(b, Real) or not isinstance(c, Real) or not isinstance(d, Real):
            raise NotImplementedError

        self.normal = Vector3(a, b, c)
        self.d = d

    def __str__(self):
        return f"[normal={self.normal}, d={self.d}]"

    def __eq__(self, other):
        if not isinstance(other, Plane):
            return NotImplemented
        return (self.normal == other.normal) and close_enough(self.d, other.d)

    def __ne__(self, other):
        if not isinstance(other, Plane):
            return NotImplemented
        return not self == other

    def classify_point(self, point):
        """
        Calculate the signed distance of the point to the plane.
        If the plane is normalized, the distance calculated is the true distance.
        Otherwise the distance is in units of the magnitude of the plane's normal vector.
        Regarless of whether the distance is true, the distance indicates the point's relationship to the plane:

        Return:
            > 0 if the point lies in front of the plane
            < 0 if the point lies behind the plane
              0 if the point lies on the plane
        """
        if not isinstance(point, Vector3):
            raise NotImplementedError

        return Vector3.dot(self.normal, point) + self.d

    def has_collided_with_box(self, box):
        """
        Determines whether the bounding box has collided with the plane.
        The algorithm used is taken from here: https://gdbooks.gitbooks.io/3dcollisions/content/Chapter2/static_aabb_plane.html
        Explanation of how the vector dot product gives the radius: https://math.stackexchange.com/a/4063364
        """
        if not isinstance(box, BoundingBox):
            raise NotImplementedError

        center = box.center()
        max_extent = box.max - center
        radius = Vector3.dot(max_extent, self.normal)
        distance = Vector3.dot(self.normal, center) + self.d
        return abs(distance) <= abs(radius)

    def has_collided_with_sphere(self, sphere):
        """Determine whether the bounding sphere has collided with the plane."""
        if not isinstance(sphere, BoundingSphere):
            raise NotImplementedError

        distance = Vector3.dot(self.normal, sphere.center) + self.d
        return abs(distance) <= sphere.radius

    def normalize(self):
        length = 1.0 / self.normal.magnitude()
        self.normal *= length
        self.d *= length

    def from_point_normal(self, point, normal):
        """Create a plane from a point on the plane and the direction the plane is facing."""
        if not isinstance(point, Vector3) or not isinstance(normal, Vector3):
            raise NotImplementedError

        self.normal.x = normal.x
        self.normal.y = normal.y
        self.normal.z = normal.z
        self.d = -Vector3.dot(normal, point)
        self.normalize()

    def from_points(self, point1, point2, point3):
        """Create a plane from three points on the plane."""
        if not isinstance(point1, Vector3) or not isinstance(point2, Vector3) or not isinstance(point3, Vector3):
            raise NotImplementedError

        self.normal = Vector3.cross(point2 - point1, point3 - point1)
        self.d = -Vector3.dot(self.normal, point1)
        self.normalize()

    @staticmethod
    def create_from_point_normal(point, normal):
        """Create a plane from a point on the plane and the direction the plane is facing."""
        result = Plane()
        result.from_point_normal(point, normal)
        return result

    @staticmethod
    def create_from_points(point1, point2, point3):
        """Create a plane from three points on the plane."""
        result = Plane()
        result.from_points(point1, point2, point3)
        return result