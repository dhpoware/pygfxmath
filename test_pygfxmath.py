# test_pygfxmath.py
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

"""Test cases for the pymathlib module."""

import math

from pygfxmath import close_enough, Vector2, Vector3, Vector4, Matrix3, Matrix4, Quaternion, BoundingBox, BoundingSphere, Plane

class TestVector2(object):
    """pytest class to unit test the pygfxmath.Vector2 class."""

    def test_distancesq(self):
        result = Vector2.distancesq(Vector2(0.0, 0.0), Vector2(2.0, 0.0))
        assert 4.0 == result, f"test_distancesq() failed, 4.0 == result:{result}"

    def test_distance(self):
        result = Vector2.distance(Vector2(0.0, 0.0), Vector2(2.0, 0.0))
        assert 2.0 == result, f"test_distance() failed, 2.0 == result:{result}"

    def test_dot(self):
        a = Vector2(3.0, 5.0)
        b = Vector2(2.0, 7.0)
        result = Vector2.dot(a, b)
        expected = 41.0
        assert close_enough(expected, result), f"test_dot() failed, expected:{expected} == result:{result}"

    def test__eq__(self):
        v = Vector2(1.0, 2.0)
        assert v == v, f"test__eq__() failed, {v} == {v}"

    def test__ne__(self):
        v1 = Vector2(1.0, 1.0)
        v2 = Vector2(2.0, 2.0)
        assert v1 != v2, f"test__ne__() failed, {v1} != {v2}"

    def test__neg__(self):
        result = -Vector2(1.0, 1.0)
        expected = Vector2(-1.0, -1.0)
        assert expected == result, f"test__neg__() failed, expected:{expected} == result:{result}"

    def test__mul__(self):
        scalar = 2.0
        vector = Vector2(1.0, 2.0)
        expected = Vector2(2.0, 4.0)
        result = scalar * vector
        assert expected == result, f"test__mul__() failed, expected:{expected} == result:{result}"

    def test__add__(self):
        result = Vector2(1.0, 2.0) + Vector2(1.0, 2.0)
        expected = Vector2(2.0, 4.0)
        assert expected == result, f"test__add__() failed, expected:{expected} == result:{result}"

    def test__sub__(self):
        result = Vector2(3.0, 3.0) - Vector2(2.0, 2.0)
        expected = Vector2(1.0, 1.0)
        assert expected == result, f"test__sub__() failed, expected:{expected} == result:{result}"

    def test__truediv__(self):
        result = Vector2(1.0, 1.0) / 2.0
        expected = Vector2(0.5, 0.5)
        assert expected == result, f"test__truediv__() failed, expected:{expected} == result:{result}"

    def test_magnitude(self):
        vector = Vector2(1.0, 0.0)
        result = vector.magnitude()
        assert 1.0 == result, f"test_magnitude() failed, 1.0 == result:{result}"

    def test_magnitudesq(self):
        vector = Vector2(1.0, 1.0)
        result = vector.magnitudesq()
        assert 2.0 == result, f"test_magnitudesq() failed, 2.0 == result:{result}"

    def test_inverse(self):
        vector = Vector2(1.0, 1.0)
        result = vector.inverse()
        expected = Vector2(-1.0, -1.0)
        assert expected == result, f"test_inverse() failed, expected:{expected} == result:{result}"

    def test_normalize(self):
        vector = Vector2(3.0, 5.0)
        vector.normalize()
        magnitude = vector.magnitude()
        assert close_enough(1.0, magnitude), f"test_normalize() failed, 1.0 == magnitude:{magnitude}"

    def test_lerp(self):
        a = Vector2(0.0, 0.0)
        b = Vector2(0.5, 0.0)
        c = Vector2(1.0, 0.0)

        # Case 1.
        result = Vector2.lerp(a, c, 0.0)
        assert a == result, f"test_lerp() case 1 failed, a:{a} == result:{result}"

        # Case 2.
        result = Vector2.lerp(a, c, 0.5)
        assert b == result, f"test_lerp() case 2 failed, b:{b} == result:{result}"

        # Case 3.
        result = Vector2.lerp(a, c, 1.0)
        assert c == result, f"test_lerp() case 3 failed, c:{c} == result:{result}"

    def test_proj(self):
        result = Vector2.proj(Vector2(2.0, 3.0), Vector2(1.0, 1.0))
        expected = Vector2(5.0/2.0, 5.0/2.0)
        assert expected == result, f"test_proj() failed, expected:{expected} == result:{result}"

    def test_perp(self):
        pass    # TODO: Can't think of an easy unit test...

    def test_reflect(self):
        i = Vector2(0.0, 0.0) - Vector2(-1.0, 1.0)
        n = Vector2(0.0, 1.0)
        result = Vector2.reflect(i, n)
        expected = Vector2(1.0, 1.0) - Vector2(0.0, 0.0)
        assert expected == result, f"test_reflect() failed, expected:{expected} == result:{result}"


class TestVector3(object):
    """pytest class to unit test the pygfxmath.Vector3 class."""

    def test__eq__(self):
        a = Vector3(1.0, 2.0, 3.0)
        b = Vector3(1.0, 2.0, 3.0)
        assert a == b, f"test__eq__() failed, a:{a} == b:{b}"

    def test__ne__(self):
        a = Vector3(1.0, 2.0, 3.0)
        b = Vector3(4.0, 5.0, 6.0)
        assert a != b, f"test__ne__() failed, a:{a} != b:{b}"

    def test__neg__(self):
        result = -Vector3(1.0, 1.0, 1.0)
        expected = Vector3(-1.0, -1.0, -1.0)
        assert expected == result, f"test__neg__() failed, expected:{expected} == result:{result}"

    def test__mul__(self):
        result = 2.0 * Vector3(1.0, 1.0, 1.0)
        expected = Vector3(2.0, 2.0, 2.0)
        assert expected == result, f"test__mul__() failed, expected:{expected} == result:{result}"

    def test__add__(self):
        a = Vector3(1.0, 2.0, 3.0)
        b = Vector3(4.0, 5.0, 6.0)
        result = a + b
        expected = Vector3(5.0, 7.0, 9.0)
        assert expected == result, f"test__add__() failed, expected:{expected} == a:{a} + b:{b}"

    def test__sub__(self):
        a = Vector3(3.0, 3.0, 3.0)
        b = Vector3(1.0, 1.0, 1.0)
        result = a - b
        expected = Vector3(2.0, 2.0, 2.0)
        assert expected == result, f"test__sub__() failed, expected:{expected} == a:{a} - b:{b}"

    def test__truediv__(self):
        result = Vector3(4.0, 4.0, 4.0) / 2.0
        expected = Vector3(2.0, 2.0, 2.0)
        assert expected == result, f"test__truediv__() failed, expected:{expected} == result:{result}"

    def test_magnitude(self):
        a = Vector3(1.0, 0.0, 0.0)
        result = a.magnitude()
        expected = 1.0
        assert close_enough(expected, result), f"test_magnitude() failed, result:{result} == expected:{expected}"

    def test_magnitudesq(self):
        a = Vector3(2.0, 0.0, 0.0)
        result = a.magnitudesq()
        expected = 4.0
        assert close_enough(expected, result), f"test_magnitudesq() failed, result:{result} == expected:{expected}"

    def test_inverse(self):
        a = Vector3(1.0, 1.0, 1.0)
        result = a.inverse()
        expected = Vector3(-1.0, -1.0, -1.0)
        assert expected == result, f"test_inverse() failed, expected:{expected} == result:{result}"

    def test_normalize(self):
        a = Vector3(1.0, 1.0, 1.0)
        a.normalize()
        result = a.magnitude()
        expected = 1.0
        assert close_enough(expected, result), f"test_normalize() failed, expected:{expected} == result:{result}"

    def test_cross(self):
        result = Vector3.cross(Vector3(1.0, 0.0, 0.0), Vector3(0.0, 1.0, 0.0))
        expected = Vector3(0.0, 0.0, 1.0)
        assert expected == result, f"test_cross() failed, expected:{expected} == result:{result}"

    def test_distance(self):
        pt1 = Vector3(0.0, 0.0, 0.0)
        pt2 = Vector3(1.0, 0.0, 0.0)
        result = Vector3.distance(pt1, pt2)
        expected = 1.0
        assert close_enough(expected, result), f"test_distance() failed, expected:{expected} == result:{result}"

    def test_distancesq(self):
        pt1 = Vector3(0.0, 0.0, 0.0)
        pt2 = Vector3(2.0, 0.0, 0.0)
        result = Vector3.distancesq(pt1, pt2)
        expected = 4.0
        assert close_enough(expected, result), f"test_distancesq() failed, expected:{expected} == result:{result}"

    def test_dot(self):
        a = Vector3(3.0, 5.0, 8.0)
        b = Vector3(2.0, 7.0, 1.0)
        result = Vector3.dot(a, b)
        expected = 49
        assert close_enough(expected, result), f"test_dot() failed, expected:{expected} == result:{result}"

    def test_lerp(self):
        a = Vector3(0.0, 0.0, 0.0)
        b = Vector3(0.0, 0.0, 0.5)
        c = Vector3(0.0, 0.0, 1.0)

        # Case 1.
        result = Vector3.lerp(a, c, 0.0)
        assert a == result, f"test_lerp() case 1 failed, a:{a} == result:{result}"

        # Case 2.
        result = Vector3.lerp(a, c, 0.5)
        assert b == result, f"test_lerp() case 2 failed, b:{b} == result:{result}"

        # Case 3.
        result = Vector3.lerp(a, c, 1.0)
        assert c == result, f"test_lerp() case 3 failed, c:{c} == result:{result}"

    def test_proj(self):
        result = Vector3.proj(Vector3(1.0, -2.0, 0.0), Vector3(2.0, 2.0, 1.0))
        expected = Vector3(-4.0/9.0, -4.0/9.0, -2.0/9.0)
        assert expected == result, f"test_proj() failed, expected:{expected} == result:{result}"

    def test_perp(self):
        pass    # TODO: Can't think of an easy unit test...

    def test_reflect(self):
        i = Vector3(0.0, 0.0, 0.0) - Vector3(-1.0, 1.0, 0.0)
        n = Vector3(0.0, 1.0, 0.0)
        result = Vector3.reflect(i, n)
        expected = Vector3(1.0, 1.0, 0.0) - Vector3(0.0, 0.0, 0.0)
        assert expected == result, f"test_reflect() failed, expected:{expected} == result:{result}"

    def test_orthogonalize2(self):
        x_axis = Vector3(1.0, 0.0, 0.0)
        y_axis = Vector3(0.0, 1.0, 0.0)
        theta = math.radians(45.0)
        axis = Vector3(math.cos(theta), math.sin(theta), 0.0)
        result_x_axis, result_y_axis = Vector3.orthogonalize2(x_axis, axis)
        assert x_axis == result_x_axis and y_axis == result_y_axis, f"test_orthogonalize2() failed, result_x_axis:{result_x_axis} result_y_axis:{result_y_axis}"

    def test_orthogonalize3(self):
        x_axis = Vector3(1.0, 0.0, 0.0)
        y_axis = Vector3(0.0, 1.0, 0.0)
        z_axis = Vector3(0.0, 0.0, 1.0)
        theta = math.radians(45.0)
        axis1 = Vector3(math.cos(theta), math.sin(theta), 0.0)
        axis2 = Vector3(math.cos(theta), 0.0, math.sin(theta))
        result_x_axis, result_y_axis, result_z_axis = Vector3.orthogonalize3(x_axis, axis1, axis2);
        assert x_axis == result_x_axis and y_axis == result_y_axis and z_axis == result_z_axis, f"test_orthogonalize3() failed, result_x_axis:{result_x_axis} result_y_axis:{result_y_axis} result_z_axis:{result_z_axis}"


class TestVector4(object):
    """pytest class to unit test the pygfxmath.Vector4 class."""

    def test__eq__(self):
        a = Vector4(1.0, 2.0, 3.0, 4.0)
        b = Vector4(1.0, 2.0, 3.0, 4.0)
        assert a == b, f"test__eq__() failed, a:{a} == b:{b}"

    def test__ne__(self):
        a = Vector4(1.0, 2.0, 3.0, 4.0)
        b = Vector4(5.0, 6.0, 7.0, 8.0)
        assert a != b, f"test__ne__() failed, a:{a} != b:{b}"

    def test__neg__(self):
        result = -Vector4(1.0, 1.0, 1.0, 1.0)
        expected = Vector4(-1.0, -1.0, -1.0, -1.0)
        assert expected == result, f"test__neg__() failed, expected:{expected} == result:{result}"

    def test__mul__(self):
        result = 2.0 * Vector4(1.0, 1.0, 1.0, 1.0)
        expected = Vector4(2.0, 2.0, 2.0, 2.0)
        assert expected == result, f"test__mul__() failed, expected:{expected} == result:{result}"

    def test__add__(self):
        a = Vector4(1.0, 2.0, 3.0, 4.0)
        b = Vector4(5.0, 6.0, 7.0, 8.0)
        result = a + b
        expected = Vector4(6.0, 8.0, 10.0, 12.0)
        assert expected == result, f"test__add__() failed, expected:{expected} == a:{a} + b:{b}"

    def test__sub__(self):
        a = Vector4(3.0, 3.0, 3.0, 3.0)
        b = Vector4(1.0, 1.0, 1.0, 1.0)
        result = a - b
        expected = Vector4(2.0, 2.0, 2.0, 2.0)
        assert expected == result, f"test__sub__() failed, expected:{expected} == a:{a} - b:{b}"

    def test__truediv__(self):
        result = Vector4(4.0, 4.0, 4.0, 4.0) / 2.0
        expected = Vector4(2.0, 2.0, 2.0, 2.0)
        assert expected == result, f"test__truediv__() failed, expected:{expected} == result:{result}"

    def test_magnitude(self):
        a = Vector4(1.0, 0.0, 0.0, 0.0)
        result = a.magnitude()
        expected = 1.0
        assert close_enough(expected, result), f"test_magnitude() failed, result:{result} == expected:{expected}"

    def test_magnitudesq(self):
        a = Vector4(2.0, 0.0, 0.0, 0.0)
        result = a.magnitudesq()
        expected = 4.0
        assert close_enough(expected, result), f"test_magnitudesq() failed, result:{result} == expected:{expected}"

    def test_inverse(self):
        a = Vector4(1.0, 1.0, 1.0, 1.0)
        result = a.inverse()
        expected = Vector4(-1.0, -1.0, -1.0, -1.0)
        assert expected == result, f"test_inverse() failed, expected:{expected} == result:{result}"

    def test_normalize(self):
        a = Vector4(1.0, 1.0, 1.0, 1.0)
        a.normalize()
        result = a.magnitude()
        expected = 1.0
        assert close_enough(expected, result), f"test_normalize() failed, expected:{expected} == result:{result}"

    def test_to_vector3(self):
        # Case 1.
        vec4 = Vector4(2.0, 4.0, 6.0, 0.0)
        result = vec4.to_vector3()
        expected = Vector3(2.0, 4.0, 6.0)
        assert expected == result, f"test_to_vector3() case 1 failed, expected:{expected} == result:{result}"

        # Case 2.
        vec4 = Vector4(2.0, 4.0, 6.0, 2.0)
        result = vec4.to_vector3()
        expected = Vector3(1.0, 2.0, 3.0)
        assert expected == result, f"test_to_vector3() case 2 failed, expected:{expected} == result:{result}"

    def test_from_vector3(self):
        result = Vector4.from_vector3(Vector3(1.0, 2.0, 3.0), 0.0)
        expected = Vector4(1.0, 2.0, 3.0, 0.0)
        assert expected == result, f"test_from_vector3() failed, expected:{expected} == result:{result}"

    def test_distancesq(self):
        pt1 = Vector4(0.0, 0.0, 0.0, 1.0)
        pt2 = Vector4(1.0, 0.0, 0.0, 1.0)
        result = Vector4.distancesq(pt1, pt2)
        expected = 1.0
        assert close_enough(expected, result), f"test_distancesq() failed, expected:{expected} == result:{result}"

    def test_distance(self):
        pt1 = Vector4(0.0, 0.0, 0.0, 1.0)
        pt2 = Vector4(4.0, 0.0, 0.0, 1.0)
        result = Vector4.distance(pt1, pt2)
        expected = 4.0
        assert close_enough(expected, result), f"test_distance() failed, expected:{expected} == result:{result}"

    def test_dot(self):
        a = Vector4(3.0, 5.0, 8.0, 0.0)
        b = Vector4(2.0, 7.0, 1.0, 0.0)
        result = Vector4.dot(a, b)
        expected = 49.0
        assert close_enough(expected, result), f"test_dot() failed, expected:{expected} == result:{result}"

    def test_lerp(self):
        a = Vector4(0.0, 0.0, 0.0, 1.0)
        b = Vector4(0.0, 0.0, 0.5, 1.0)
        c = Vector4(0.0, 0.0, 1.0, 1.0)

        # Case 1.
        result = Vector4.lerp(a, c, 0.0)
        assert a == result, f"test_lerp() case 1 failed, a:{a} == result:{result}"

        # Case 2.
        result = Vector4.lerp(a, c, 0.5)
        assert b == result, f"test_lerp() case 2 failed, b:{b} == result:{result}"

        # Case 3.
        result = Vector4.lerp(a, c, 1.0)
        assert c == result, f"test_lerp() case 3 failed, c:{c} == result:{result}"


class TestMatrix3(object):
    """pytest class to unit test the pygfxmath.Matrix3 class."""

    def test__eq__(self):
        a = Matrix3.identity()
        b = Matrix3.identity()
        assert a == b, f"test__eq__() failed, a:{a} == b:{b}"

    def test__ne__(self):
        a = Matrix3([1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0])
        b = Matrix3([2.0, 2.0, 2.0], [2.0, 2.0, 2.0], [2.0, 2.0, 2.0])
        assert a != b, f"test__ne__() failed, a:{a} != b:{b}"

    def test__neg__(self):
        result = -Matrix3.identity()
        expected = Matrix3([-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0])
        assert expected == result, f"test__neg__() failed, expected:{expected} == result:{result}"

    def test__mul__(self):
        # Case 1: Multiply by scalar.
        result = Matrix3.identity() * 2.0
        expected = Matrix3([2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0])
        assert expected == result, f"test__mul__() case 1 failed, expected:{expected} == result:{result}"

        # Case 2: Multiply by matrix.
        a = Matrix3([1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0])
        b = Matrix3([2.0, 2.0, 2.0], [2.0, 2.0, 2.0], [2.0, 2.0, 2.0])
        result = a * b
        expected = Matrix3([6.0, 6.0, 6.0], [6.0, 6.0, 6.0], [6.0, 6.0, 6.0])
        assert expected == result, f"test__mul__() case 2 failed, expected:{expected} == result:{result}"

        # Case 3: Multiply by vector.
        v = Vector3(1.0, 2.0, 3.0)
        m = Matrix3([2.0, 2.0, 2.0], [2.0, 2.0, 2.0], [2.0, 2.0, 2.0])
        result = v * m
        expected = Vector3(12.0, 12.0, 12.0)
        assert expected == result, f"test__mul__() case 3 failed, expected:{expected} == result:{result}"

    def test__add__(self):
        result = Matrix3.identity() + Matrix3.identity()
        expected = Matrix3([2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0])
        assert expected == result, f"test__add__() failed, expected:{expected} == result:{result}"

    def test__sub__(self):
        result = Matrix3.identity() - Matrix3.identity()
        expected = Matrix3()
        assert expected == result, f"test__sub__() failed, expected:{expected} == result:{result}"

    def test__truediv__(self):
        result = Matrix3([2.0, 2.0, 2.0], [2.0, 2.0, 2.0], [2.0, 2.0, 2.0]) / 2.0
        expected = Matrix3([1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0])
        assert expected == result, f"test__truediv__() failed, expected:{expected} == result:{result}"

    def test_determinant(self):
        m = Matrix3([2.0, 1.0, 3.0], [4.0, 0.0, 1.0], [2.0, -1.0, 2.0])
        result = m.determinant()
        expected = -16.0
        assert close_enough(expected, result), f"test_determinant() failed, expected:{expected} == result:{result}"

    def test_transpose(self):
        m = Matrix3([1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0])
        result = m.transpose()
        expected = Matrix3([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
        assert expected == result, f"test_transpose() failed, expected:{expected} == result:{result}"

    def test_orient(self):
        # Case 1: Rotate the x-axis into the x-axis.
        from_vector = Vector3(1.0, 0.0, 0.0)
        to_vector= Vector3(1.0, 0.0, 0.0)
        rotation_matrix = Matrix3.create_orient(from_vector, to_vector)
        result = from_vector * rotation_matrix
        expected = to_vector
        assert expected == result, f"test_orient() case 1 failed, expected:{expected} == result:{result}"

        # Case 2: Rotate the x-axis into the negative x-axis.
        from_vector = Vector3(1.0, 0.0, 0.0)
        to_vector = Vector3(-1.0, 0.0, 0.0)
        rotation_matrix = Matrix3.create_orient(from_vector, to_vector)
        result = from_vector * rotation_matrix
        expected = to_vector
        assert expected == result, f"test_orient() case 2 failed, expected:{expected} == result:{result}"

        # Case 3: Rotate the x-axis into the y-axis.
        from_vector = Vector3(1.0, 0.0, 0.0)
        to_vector = Vector3(0.0, 1.0, 0.0)
        rotation_matrix = Matrix3.create_orient(from_vector, to_vector)
        result = from_vector * rotation_matrix
        expected = to_vector
        assert expected == result, f"test_orient() case 3 failed, expected:{expected} == result:{result}"

    def test_rotate(self):
        # Rotations are orthogonal - i.e., the transpose of a rotation matrix is the same as the inverse of the rotation matrix.
        # Thus an inverse always exists, and so the determinant of the rotation matrix cannot be zero.
        m = Matrix3.create_rotate(Vector3(1.0, 0.0, 0.0), 90.0)
        assert not close_enough(m.determinant(), 0.0), f"test_rotate() failed, determinant of a rotation matrix cannot be zero"

    def test_to_axes(self):
        m = Matrix3([0.707, -0.707, 0.0], [1.250, 1.250, 0.0], [0.0, 0.0, 1.0])
        result_x_axis, result_y_axis, result_z_axis = m.to_axes()
        expected_x_axis = Vector3(0.707, -0.707, 0.0)
        expected_y_axis = Vector3(1.250, 1.250, 0.0)
        expected_z_axis = Vector3(0.0, 0.0, 1.0)
        assert expected_x_axis == result_x_axis, "test_to_axes() failed, expected_x_axis:{expected_x_axis} == result_x_axis:{result_x_axis}"
        assert expected_y_axis == result_y_axis, "test_to_axes() failed, expected_y_axis:{expected_y_axis} == result_y_axis:{result_y_axis}"
        assert expected_z_axis == result_z_axis, "test_to_axes() failed, expected_z_axis:{expected_z_axis} == result_z_axis:{result_z_axis}"

    def test_to_axes_transposed(self):
        m = Matrix3([0.707, -0.707, 0.0], [1.250, 1.250, 0.0], [0.0, 0.0, 1.0])
        result_x_axis, result_y_axis, result_z_axis = m.to_axes_transposed()
        expected_x_axis = Vector3(0.707, 1.250, 0.0)
        expected_y_axis = Vector3(-0.707, 1.250, 0.0)
        expected_z_axis = Vector3(0.0, 0.0, 1.0)
        assert expected_x_axis == result_x_axis, "test_to_axes_transposed() failed, expected_x_axis:{expected_x_axis} == result_x_axis:{result_x_axis}"
        assert expected_y_axis == result_y_axis, "test_to_axes_transposed() failed, expected_y_axis:{expected_y_axis} == result_y_axis:{result_y_axis}"
        assert expected_z_axis == result_z_axis, "test_to_axes_transposed() failed, expected_z_axis:{expected_z_axis} == result_z_axis:{result_z_axis}"

    def test_heading_pitch_roll(self):
        euler_in = (10.0, 20.0, 30.0)
        m = Matrix3.create_from_heading_pitch_roll(euler_in[0], euler_in[1], euler_in[2])
        result_heading, result_pitch, result_roll = m.to_heading_pitch_roll()
        expected_heading = euler_in[0]
        expected_pitch = euler_in[1]
        expected_roll = euler_in[2]
        assert close_enough(expected_heading, result_heading), f"test_heading_pitch_roll() failed, expected_heading:{expected_heading} == result_heading:{result_heading}"
        assert close_enough(expected_pitch, result_pitch), f"test_heading_pitch_roll() failed, expected_pitch:{expected_pitch} == result_pitch:{result_pitch}"
        assert close_enough(expected_roll, result_roll), f"test_heading_pitch_roll() failed, expected_roll:{expected_roll} == result_roll:{result_roll}"

    def test_inverse(self):
        m = Matrix3.create_from_heading_pitch_roll(10.0, 20.0, 30.0)
        result = m * m.inverse()
        expected = Matrix3.identity()
        assert expected == result, f"test_inverse() failed, expected:{expected} == result:{result}"


class TestMatrix4(object):
    """pytest class to unit test the pygfxmath.Matrix4 class."""

    def test__eq__(self):
        a = Matrix4.identity()
        b = Matrix4.identity()
        assert a == b, f"test__eq__() failed, a:{a} == b:{b}"

    def test__ne__(self):
        a = Matrix4([1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0])
        b = Matrix4([2.0, 2.0, 2.0, 2.0], [2.0, 2.0, 2.0, 2.0], [2.0, 2.0, 2.0, 2.0], [2.0, 2.0, 2.0, 2.0])
        assert a != b, f"test__ne__() failed, a:{a} != b:{b}"

    def test__neg__(self):
        result = -Matrix4.identity()
        expected = Matrix4([-1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.0], [0.0, 0.0, 0.0, -1.0])
        assert expected == result, f"test__neg__() failed, expected:{expected} == result:{result}"

    def test__mul__(self):
        # Case 1: Multiply by scalar.
        result = Matrix4.identity() * 2.0
        expected = Matrix4([2.0, 0.0, 0.0, 0.0], [0.0, 2.0, 0.0, 0.0], [0.0, 0.0, 2.0, 0.0], [0.0, 0.0, 0.0, 2.0])
        assert expected == result, f"test__mul__() case 1 failed, expected:{expected} == result:{result}"

        # Case 2: Multiply by matrix.
        a = Matrix4([1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0])
        b = Matrix4([2.0, 2.0, 2.0, 2.0], [2.0, 2.0, 2.0, 2.0], [2.0, 2.0, 2.0, 2.0], [2.0, 2.0, 2.0, 2.0])
        result = a * b
        expected = Matrix4([8.0, 8.0, 8.0, 8.0], [8.0, 8.0, 8.0, 8.0], [8.0, 8.0, 8.0, 8.0], [8.0, 8.0, 8.0, 8.0])
        assert expected == result, f"test__mul__() case 2 failed, expected:{expected} == result:{result}"

        # Case 3: Multiply by vector.
        v = Vector4(1.0, 2.0, 3.0, 4.0)
        m = Matrix4([2.0, 2.0, 2.0, 2.0], [2.0, 2.0, 2.0, 2.0], [2.0, 2.0, 2.0, 2.0], [2.0, 2.0, 2.0, 2.0])
        result = v * m
        expected = Vector4(20.0, 20.0, 20.0, 20.0)
        assert expected == result, f"test__mul__() case 3 failed, expected:{expected} == result:{result}"

    def test__add__(self):
        result = Matrix4.identity() + Matrix4.identity()
        expected = Matrix4([2.0, 0.0, 0.0, 0.0], [0.0, 2.0, 0.0, 0.0], [0.0, 0.0, 2.0, 0.0], [0.0, 0.0, 0.0, 2.0])
        assert expected == result, f"test__add__() failed, expected:{expected} == result:{result}"

    def test__sub__(self):
        result = Matrix4.identity() - Matrix4.identity()
        expected = Matrix4()
        assert expected == result, f"test__sub__() failed, expected:{expected} == result:{result}"

    def test__truediv__(self):
        result = Matrix4([2.0, 2.0, 2.0, 2.0], [2.0, 2.0, 2.0, 2.0], [2.0, 2.0, 2.0, 2.0], [2.0, 2.0, 2.0, 2.0]) / 2.0
        expected = Matrix4([1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0])
        assert expected == result, f"test__truediv__() failed, expected:{expected} == result:{result}"

    def test_determinant(self):
        m = Matrix4([2.0, 1.0, 3.0, 0.0], [4.0, 0.0, 1.0, 0.0], [2.0, -1.0, 2.0, 0.0], [0.0, 0.0, 0.0, 2.0])
        result = m.determinant()
        expected = -32.0
        assert close_enough(expected, result), f"test_determinant() failed, expected:{expected} == result:{result}"

    def test_transpose(self):
        m = Matrix4([1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 2.0, 2.0], [3.0, 3.0, 3.0, 3.0], [4.0, 4.0, 4.0, 4.0])
        result = m.transpose()
        expected = Matrix4([1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0])
        assert expected == result, f"test_transpose() failed, expected:{expected} == result:{result}"

    def test_rotate(self):
        m1 = Matrix4.create_rotate(Vector4(1.0, 0.0, 0.0, 0.0), 45.0)
        m2 = m1.inverse()
        result = m1 * m2
        expected = Matrix4.identity()
        assert expected == result, f"test_rotate() failed, expected:{expected} == result:{result}"

    def test_to_axes(self):
        m = Matrix4([0.707, -0.707, 0.0, 0.0], [1.250, 1.250, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0])
        result_x_axis, result_y_axis, result_z_axis = m.to_axes()
        expected_x_axis = Vector4(0.707, -0.707, 0.0, 0.0)
        expected_y_axis = Vector4(1.250, 1.250, 0.0, 0.0)
        expected_z_axis = Vector4(0.0, 0.0, 1.0, 0.0)
        assert expected_x_axis == result_x_axis, "test_to_axes() failed, expected_x_axis:{expected_x_axis} == result_x_axis:{result_x_axis}"
        assert expected_y_axis == result_y_axis, "test_to_axes() failed, expected_y_axis:{expected_y_axis} == result_y_axis:{result_y_axis}"
        assert expected_z_axis == result_z_axis, "test_to_axes() failed, expected_z_axis:{expected_z_axis} == result_z_axis:{result_z_axis}"

    def test_to_axes_transposed(self):
        m = Matrix4([0.707, -0.707, 0.0, 0.0], [1.250, 1.250, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0])
        result_x_axis, result_y_axis, result_z_axis = m.to_axes_transposed()
        expected_x_axis = Vector4(0.707, 1.250, 0.0, 0.0)
        expected_y_axis = Vector4(-0.707, 1.250, 0.0, 0.0)
        expected_z_axis = Vector4(0.0, 0.0, 1.0, 0.0)
        assert expected_x_axis == result_x_axis, "test_to_axes_transposed() failed, expected_x_axis:{expected_x_axis} == result_x_axis:{result_x_axis}"
        assert expected_y_axis == result_y_axis, "test_to_axes_transposed() failed, expected_y_axis:{expected_y_axis} == result_y_axis:{result_y_axis}"
        assert expected_z_axis == result_z_axis, "test_to_axes_transposed() failed, expected_z_axis:{expected_z_axis} == result_z_axis:{result_z_axis}"

    def test_heading_pitch_roll(self):
        euler_in = (10.0, 20.0, 30.0)
        m = Matrix4.create_from_heading_pitch_roll(euler_in[0], euler_in[1], euler_in[2])
        result_heading, result_pitch, result_roll = m.to_heading_pitch_roll()
        expected_heading = euler_in[0]
        expected_pitch = euler_in[1]
        expected_roll = euler_in[2]
        assert close_enough(expected_heading, result_heading), f"test_heading_pitch_roll() failed, expected_heading:{expected_heading} == result_heading:{result_heading}"
        assert close_enough(expected_pitch, result_pitch), f"test_heading_pitch_roll() failed, expected_pitch:{expected_pitch} == result_pitch:{result_pitch}"
        assert close_enough(expected_roll, result_roll), f"test_heading_pitch_roll() failed, expected_roll:{expected_roll} == result_roll:{result_roll}"


class TestQuaternion(object):
    """pytest class to unit test the pygfxmath.Quaternion class."""

    def test__eq__(self):
        a = Quaternion.identity()
        b = Quaternion.identity()
        assert a == b, f"test__eq__() failed, a:{a} == b:{b}"

    def test__ne__(self):
        a = Quaternion(1.0, 2.0, 3.0, 4.0)
        b = Quaternion(5.0, 6.0, 7.0, 8.0)
        assert a != b, f"test__ne__() failed, a:{a} != b:{b}"

    def test__add__(self):
        a = Quaternion(1.0, 1.0, 1.0, 1.0)
        result = a + a
        expected = Quaternion(2.0, 2.0, 2.0, 2.0)
        assert expected == result, f"test__add__() failed, expected:{expected} == result:{result}"

    def test__sub__(self):
        a = Quaternion(1.0, 1.0, 1.0, 1.0)
        result = a - a
        expected = Quaternion(0.0, 0.0, 0.0, 0.0)
        assert expected == result, f"test__sub__() failed, expected:{expected} == result:{result}"

    def test__mul__(self):
        # Case 1: Multiply by scalar.
        a = Quaternion(1.0, 1.0, 1.0, 1.0)
        result = a * 2.0
        expected = Quaternion(2.0, 2.0, 2.0, 2.0)
        assert expected == result, f"test__mul__() case 1 failed, expected:{expected} == result:{result}"

        # Case 2: Multiply by quaternion.
        # Rotate the x-axis 90 degrees about the z-axis so that the x-axis will become the y-axis.
        axis_in = Vector3(1.0, 0.0, 0.0)
        pure = Quaternion(0.0, axis_in.x, axis_in.y, axis_in.z)
        rot = Quaternion.create_from_axis_angle(Vector3(0.0, 0.0, 1.0), 90.0)
        result = rot.conjugate() * pure * rot
        axis_out = Vector3(result.x, result.y, result.z)
        expected = Vector3(0.0, 1.0, 0.0)
        assert expected == axis_out, f"test__mul__() case 2 failed, expected:{expected} == axis_out:{axis_out}"

    def test__truediv__(self):
        a = Quaternion(2.0, 2.0, 2.0, 2.0)
        result = a / 2.0
        expected = Quaternion(1.0, 1.0, 1.0, 1.0)
        assert expected == result, f"test__truediv__() failed, expected:{expected} == result:{result}"

    def test_conjugate(self):
        a = Quaternion(1.0, 2.0, 3.0, 4.0)
        result = a.conjugate()
        expected = Quaternion(1.0, -2.0, -3.0, -4.0)
        assert expected == result, f"test_conjugate() failed, expected:{expected} == result:{result}"

    def test_axis_angle_conversion(self):
        # Case 1: Axis and angle.
        degrees_in = 45.0
        axis_in = Vector3(0.0, 1.0, 0.0)
        q = Quaternion.create_from_axis_angle(axis_in, degrees_in)
        axis_out, degrees_out = q.to_axis_angle()
        assert close_enough(degrees_out, degrees_in) and axis_out == axis_in, f"test_axis_angle_conversion() case 1 failed, degrees_in:{degrees_in} == degrees_out:{degrees_out}, axis_in:{axis_in} == axis_out:{axis_out}"

        # Case 2: Angle is zero degrees.
        # The axis is irrelevant for this case since there is no rotation occurring.
        degrees_in = 0.0
        q.from_axis_angle(axis_in, degrees_in)
        axis_out, degrees_out = q.to_axis_angle()
        assert close_enough(degrees_in, degrees_out), f"test_axis_angle_conversion() case 2 failed, degrees_in:{degrees_in} == degrees_out:{degrees_out}"

    def test_euler_conversion(self):
        euler_in = (10.0, 20.0, 30.0)
        q = Quaternion.create_from_heading_pitch_roll(euler_in[0], euler_in[1], euler_in[2])
        euler_out = q.to_heading_pitch_roll()
        assert close_enough(euler_in[0], euler_out[0]), f"test_euler_conversion() failed, euler_in[0]:{euler_in[0]} == euler_out[0]:{euler_out[0]}"
        assert close_enough(euler_in[1], euler_out[1]), f"test_euler_conversion() failed, euler_in[1]:{euler_in[1]} == euler_out[0]:{euler_out[1]}"
        assert close_enough(euler_in[2], euler_out[2]), f"test_euler_conversion() failed, euler_in[2]:{euler_in[2]} == euler_out[0]:{euler_out[2]}"

    def test_magnitude(self):
        result = Quaternion.identity().magnitude()
        expected = 1.0
        assert close_enough(expected, result), f"test_magnitude() failed, expected:{expected} == result:{result}"

    def test_normalize(self):
        a = Quaternion(2.0, 2.0, 2.0, 2.0)
        a.normalize()
        result = a.magnitude()
        expected = 1.0
        assert close_enough(expected, result), f"test_normalize() failed, expected:{expected} == result:{result}"

    def test_matrix_conversion(self):
        # Rotate the x-axis 90 degrees about the z-axis so that the x-axis will become the y-axis.
        x_axis = Vector3(1.0, 0.0, 0.0)
        y_axis = Vector3(0.0, 1.0, 0.0)
        z_axis = Vector3(0.0, 0.0, 1.0)
        q = Quaternion.create_from_axis_angle(z_axis, 90.0)
        m = q.to_matrix3()
        result = x_axis * m
        assert y_axis == result, f"test_matrix_conversion() failed, y_axis:{y_axis} == result:{result}"

    def test_point_rotation(self):
        # Rotate the point at (0,0,20) 180 degrees about the y-axis to give the transformed point (0,0,-20).
        point = Vector3(0.0, 0.0, 20.0)
        expected = Vector3(0.0, 0.0, -20.0)
        pure = Quaternion(0.0, point.x, point.y, point.z)
        rotation = Quaternion(0.0, 0.0, 1.0, 0.0)   # 180 degrees rotation around the y-axis
        result = rotation.conjugate() * pure * rotation
        transformed_point = Vector3(result.x, result.y, result.z)
        assert expected == transformed_point, f"test_point_rotation() failed, expected:{expected} == transformed_point:{transformed_point}"

    def test_slerp(self):
        # 90 degrees rotation around x-axis
        src = Quaternion(math.sqrt(0.5), math.sqrt(0.5), 0.0, 0.0)

        # -90 degrees rotation around x-axis
        dest = Quaternion(math.sqrt(0.5), -math.sqrt(0.5), 0.0, 0.0)

        # Case 1: When t == 0, result == src.
        result = Quaternion.slerp(src, dest, 0.0)
        assert result == src, f"test_slerp() case 1 failed, result:{result} == src:{src}"

        # Case 2: When t == 1, result == dest.
        result = Quaternion.slerp(src, dest, 1.0)
        assert result == dest, f"test_slerp() case 2 failed, result:{result} == src:{src}"

        # Case 3: When t == 0.5, result == 0 degrees rotation around x-axis (i.e., identity).
        result = Quaternion.slerp(src, dest, 0.5)
        assert result == Quaternion.identity(), f"test_slerp() case 3 failed, result:{result} == identity:{Quaternion.identity()}"


class TestBoundingBox(object):
    """pytest class to unit test the pygfxmath.BoundingBox class."""

    def test_center(self):
        a = BoundingBox(Vector3(-0.5, -0.5, -0.5), Vector3(0.5, 0.5, 0.5))
        result = a.center()
        expected = Vector3(0.0, 0.0, 0.0)
        assert expected == result, f"test_center() failed, expected:{expected} == result:{result}"

    def test_size(self):
        a = BoundingBox(Vector3(-0.5, -0.5, -0.5), Vector3(0.5, 0.5, 0.5))
        result = a.size()
        expected = math.sqrt(3.0)
        assert close_enough(expected, result), f"test_size() failed, expected:{expected} == result:{result}"

    def test_radius(self):
        a = BoundingBox(Vector3(-0.5, -0.5, -0.5), Vector3(0.5, 0.5, 0.5))
        result = a.radius()
        expected = math.sqrt(3.0) * 0.5
        assert close_enough(expected, result), f"test_radius() failed, expected:{expected} == result:{result}"

    def test_has_collided(self):
        # Case 1: Collision has occurred.
        a = BoundingBox(Vector3(-0.5, -0.5, -0.5), Vector3(0.5, 0.5, 0.5))
        b = BoundingBox(Vector3(-1.0, -1.0, -1.0), Vector3(1.0, 1.0, 1.0))
        result = a.has_collided(b)
        expected = True
        assert expected == result, f"test_has_collided() case 1 failed, expected:{expected} == result:{result}"

        # Case 2: No collision has occurred.
        a = BoundingBox(Vector3(10.0, 0.0, -10.0), Vector3(20.0, 10.0, 10.0))
        b = BoundingBox(Vector3(-1.0, -1.0, -1.0), Vector3(1.0, 1.0, 1.0))
        result = a.has_collided(b)
        expected = False
        assert expected == result, f"test_has_collided() case 2 failed, expected:{expected} == result:{result}"

    def test_point_in(self):
        # Case 1: Point inside bounding box.
        box = BoundingBox(Vector3(-0.5, -0.5, -0.5), Vector3(0.5, 0.5, 0.5))
        point = Vector3(0.0, 0.0, 0.0)
        result = box.point_in(point)
        expected = True
        assert expected == result, f"test_point_in() case 1 failed, expected:{expected} == result:{result}"

        # Case 2: Point outside bounding box.
        box = BoundingBox(Vector3(-0.5, -0.5, -0.5), Vector3(0.5, 0.5, 0.5))
        point = Vector3(10.0, 10.0, 10.0)
        result = box.point_in(point)
        expected = False
        assert expected == result, f"test_point_in() case 2 failed, expected:{expected} == result:{result}"


class TestBoundingSphere(object):
    """pytest class to unit test the pygfxmath.BoundingSphere class."""

    def test_has_collided(self):
        # Case 1: Collision has occurred.
        a = BoundingSphere(Vector3(0.0, 0.0, 0.0), 10.0)
        b = BoundingSphere(Vector3(5.0, 0.0, 0.0), 10.0)
        result = a.has_collided(b)
        expected = True
        assert expected == result, f"test_has_collided() case 1 failed, expected:{expected} == result:{result}"

        # Case 2: No collision has occurred.
        a = BoundingSphere(Vector3(0.0, 0.0, 0.0), 1.0)
        b = BoundingSphere(Vector3(5.0, 0.0, 0.0), 1.0)
        result = a.has_collided(b)
        expected = False
        assert expected == result, f"test_has_collided() case 2 failed, expected:{expected} == result:{result}"

    def test_point_in(self):
        # Case 1: Point inside bounding sphere.
        sphere = BoundingSphere(Vector3(0.0, 0.0, 0.0), 10.0)
        point = Vector3(0.0, 0.0, 0.0)
        result = sphere.point_in(point)
        expected = True
        assert expected == result, f"test_point_in() case 1 failed, expected:{expected} == result:{result}"

        # Case 2: Point outside bounding sphere.
        sphere = BoundingSphere(Vector3(0.0, 0.0, 0.0), 10.0)
        point = Vector3(20.0, 20.0, 20.0)
        result = sphere.point_in(point)
        expected = False
        assert expected == result, f"test_point_in() case 2 failed, expected:{expected} == result:{result}"


class TestPlane(object):
    """pytest class to unit test the pygfxmath.Plane class."""

    def test__eq__(self):
        a = Plane(0.0, 1.0, 0.0, 0.0)
        assert a == a, f"test__eq__() failed, a:{a} == a:{a}"

    def test_from_point_normal(self):
        result = Plane.create_from_point_normal(Vector3(0.0, 0.0, 0.0), Vector3(0.0, 0.0, 1.0))
        expected = Plane(0.0, 0.0, 1.0, 0.0)
        assert expected == result, f"test_from_point_normal() failed, expected:{expected} == result:{result}"

    def test_from_points(self):
        points = (Vector3(0.0,  1.0, 0.0), Vector3(-1.0, -1.0, 0.0), Vector3(1.0, -1.0, 0.0))
        result = Plane.create_from_points(points[0], points[1], points[2])
        expected = Plane(0.0, 0.0, 1.0, 0.0)
        assert expected == result, f"test_from_points() failed, expected:{expected} == result:{result}"

    def test_normalize(self):
        result = Plane(0.0, 0.0, 2.0, 0.0)
        result.normalize()
        expected = Plane(0.0, 0.0, 1.0, 0.0)
        assert expected == result, f"test_normalize() failed, expected:{expected} == result:{result}"

    def test_classify_point(self):
        plane = Plane(0.0, 0.0, 1.0, 0.0)

        # Case 1: Point on plane.
        result = plane.classify_point(Vector3(0.0, 0.0, 0.0))
        assert close_enough(result, 0.0), f"test_classify_point() case 1 failed, result:{result} == 0.0"

        # Case 2: Point in front of plane.
        result = plane.classify_point(Vector3(0.0, 0.0, 1.0))
        assert result > 0.0, f"test_classify_point() case 2 failed, result:{result} > 0.0"

        # Case 3: Point behind plane.
        result = plane.classify_point(Vector3(0.0, 0.0, -1.0))
        assert result < 0.0, f"test_classify_point() case 3 failed, result:{result} < 0.0"

    def test_has_collided_with_sphere(self):
        plane = Plane.create_from_point_normal(Vector3(0.0, 0.0, 0.0), Vector3(0.0, 1.0, 0.0))

        # Case 1: Bounding sphere hasn't collided with plane
        sphere = BoundingSphere(Vector3(0.0, 5.0, 0.0), 1.0)
        result = plane.has_collided_with_sphere(sphere)
        expected = False
        assert expected == result, f"test_has_collided() case 1 failed, expected:{expected} == result:{result}"

        # Case 2: Bounding sphere has collided with plane
        sphere = BoundingSphere(Vector3(0.0, 0.0, 0.0), 1.0)
        result = plane.has_collided_with_sphere(sphere)
        expected = True
        assert expected == result, f"test_has_collided() case 2 failed, expected:{expected} == result:{result}"

    def test_has_collided_with_box(self):
        plane = Plane.create_from_point_normal(Vector3(0.0, 0.0, 0.0), Vector3(0.0, 1.0, 0.0))

        # Case 1: Bounding box hasn't collided with plane
        box = BoundingBox(Vector3(-2.0, 10.0, -2.0), Vector3(2.0, 12.0, 2.0))
        result = plane.has_collided_with_box(box)
        expected = False
        assert expected == result, f"test_has_collided_with_box() case 1 failed, expected:{expected} == result:{result}"

        # Case 2: Bounding box has collided with plane
        box = BoundingBox(Vector3(-2.0, -2.0, -2.0), Vector3(2.0, 2.0, 2.0))
        result = plane.has_collided_with_box(box)
        expected = True
        assert expected == result, f"test_has_collided_with_box() case 2 failed, expected:{expected} == result:{result}"