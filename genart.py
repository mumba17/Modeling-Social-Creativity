import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Union, Callable, Tuple, Any
from PIL import Image
import random
import operator
from enum import Enum, auto
import logging
import os
from datetime import datetime
import json
import torch
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OpType(Enum):
    CONSTANT = auto()
    UNARY = auto()
    BINARY = auto()
    COORDINATE = auto()

@dataclass
class QuaternionTensor:
    """Quaternion implementation using PyTorch tensors"""
    data: torch.Tensor  # shape [..., 4] for [w,x,y,z]

    @property
    def device(self):
        return self.data.device

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor) -> 'QuaternionTensor':
        if tensor.shape[-1] != 4:
            raise ValueError("Last dimension must be 4 for quaternion values")
        return cls(tensor)

    def to(self, device: torch.device) -> 'QuaternionTensor':
        return QuaternionTensor(self.data.to(device))

    @property
    def w(self) -> torch.Tensor:
        return self.data[..., 0]
    
    @property
    def x(self) -> torch.Tensor:
        return self.data[..., 1]

    @property 
    def y(self) -> torch.Tensor:
        return self.data[..., 2]

    @property
    def z(self) -> torch.Tensor:
        return self.data[..., 3]

    def __add__(self, other: 'QuaternionTensor') -> 'QuaternionTensor':
        return QuaternionTensor(self.data + other.data)

    def __sub__(self, other: 'QuaternionTensor') -> 'QuaternionTensor':
        return QuaternionTensor(self.data - other.data)

    def __mul__(self, other: 'QuaternionTensor') -> 'QuaternionTensor':
        w1,x1,y1,z1 = self.w, self.x, self.y, self.z
        w2,x2,y2,z2 = other.w, other.x, other.y, other.z
        
        return QuaternionTensor(torch.stack([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ], dim=-1))

    def __truediv__(self, other: 'QuaternionTensor') -> 'QuaternionTensor':
        denom = other.norm_squared().unsqueeze(-1)
        mask = denom > 1e-10
        recip = other.conjugate()
        
        # Safe division with mask
        safe_denom = torch.where(mask, denom, torch.ones_like(denom))
        result = self * QuaternionTensor(recip.data / safe_denom)
        
        # Zero out results where denominator was too small
        result.data = torch.where(mask.unsqueeze(-1), result.data, torch.zeros_like(result.data))
        return result

    def __neg__(self) -> 'QuaternionTensor':
        return QuaternionTensor(-self.data)

    def rotate(self, angle: float) -> 'QuaternionTensor':
        c = torch.cos(torch.tensor(angle/2, device=self.device))
        s = torch.sin(torch.tensor(angle/2, device=self.device))
        return QuaternionTensor(torch.stack([
            c*self.w - s*self.x,
            c*self.x + s*self.w,
            c*self.y - s*self.z,
            c*self.z + s*self.y
        ], dim=-1))

    def floor(self) -> 'QuaternionTensor':
        return QuaternionTensor(torch.floor(self.data))

    def modulo(self, mod: float) -> 'QuaternionTensor':
        return QuaternionTensor(torch.remainder(self.data, mod))

    def conjugate(self) -> 'QuaternionTensor':
        return QuaternionTensor(torch.stack([
            self.w,
            -self.x,
            -self.y,
            -self.z
        ], dim=-1))

    def norm_squared(self) -> torch.Tensor:
        return torch.sum(self.data * self.data, dim=-1)

    def normalize(self) -> 'QuaternionTensor':
        norm = torch.sqrt(self.norm_squared()).unsqueeze(-1)
        mask = norm > 1e-10
        
        # Safe normalization with mask
        safe_norm = torch.where(mask, norm, torch.ones_like(norm))
        result = QuaternionTensor(self.data / safe_norm)
        
        # Zero out results where norm was too small
        result.data = torch.where(mask, result.data, torch.zeros_like(result.data))
        return result

    def to_rgb(self) -> torch.Tensor:
        #print(f"Converting to RGB: Input shape {self.data.shape}, dtype: {self.data.dtype}")
        if self.data.shape[-1] != 4:
            raise ValueError(f"Quaternion tensor must have 4 components, got {self.data.shape[-1]}")

        values = torch.clamp(self.data[..., 1:4], -10.0, 10.0)
        rgb = 255.0 / (1.0 + torch.exp(-values))

        if len(rgb.shape) < 3 or rgb.shape[-1] != 3:
            raise ValueError(f"RGB tensor must have 3 channels, got shape {rgb.shape}")

        #print(f"RGB shape: {rgb.shape}, dtype: {rgb.dtype}")
        return rgb.to(torch.uint8)


# Constants - defined using PyTorch tensors
GOLDEN_RATIO = (1 + torch.sqrt(torch.tensor(5.0))).clone().detach() / 2

def q_sin(q: QuaternionTensor) -> QuaternionTensor:
    return QuaternionTensor(torch.sin(q.data))

def q_exp(q: QuaternionTensor) -> QuaternionTensor:
    # Safe exponential with clamping
    return QuaternionTensor(torch.exp(torch.clamp(q.data, -30.0, 30.0)))

def q_log(q: QuaternionTensor) -> QuaternionTensor:
    # Avoid invalid log inputs
    safe_data = torch.where(q.data > 1e-10, q.data, torch.tensor(1e-10, device=q.device))
    return QuaternionTensor(torch.log(safe_data))

def q_sqrt(q: QuaternionTensor) -> QuaternionTensor:
    # Avoid negative inputs for sqrt
    safe_data = torch.where(q.data >= 0, q.data, torch.tensor(0.0, device=q.device))
    return QuaternionTensor(torch.sqrt(safe_data))

def q_cos(q: QuaternionTensor) -> QuaternionTensor:
    return QuaternionTensor(torch.cos(q.data))

def q_tan(q: QuaternionTensor) -> QuaternionTensor:
    return q_sin(q) * q_cos(q).conjugate()

def q_abs(q: QuaternionTensor) -> QuaternionTensor:
    abs_data = torch.abs(q.data)
    return QuaternionTensor(abs_data.reshape(q.data.shape))  # Ensure shape consistency

def q_power(q: QuaternionTensor) -> QuaternionTensor:
    """Raise quaternion to real power"""
    #Random Power
    p = torch.rand(1).item()
    p_tensor = torch.tensor(p, device=q.device)
    return q_exp(q_log(q) * QuaternionTensor(torch.full_like(q.data, p_tensor)))

def q_inverse(q: QuaternionTensor) -> QuaternionTensor:
    norm_squared = q.norm_squared().unsqueeze(-1)
    mask = norm_squared > 1e-10
    recip = q.conjugate()
    safe_norm = torch.where(mask, norm_squared, torch.ones_like(norm_squared))
    result = QuaternionTensor(recip.data / safe_norm)

    # Prevent additional dimensions
    result.data = result.data.squeeze() if result.data.ndim > 3 else result.data
    return result


def q_cube(q: QuaternionTensor) -> QuaternionTensor:
    result = q * q * q
    result.data = torch.clamp(result.data, -1e3, 1e3)  # Prevent overflow
    if result.data.ndim > 3:  # Avoid extra dimensions
        result.data = result.data.squeeze()
    return result

def q_sinh(q: QuaternionTensor) -> QuaternionTensor:
    """Hyperbolic sine"""
    return (q_exp(q) - q_exp(-q)) * QuaternionTensor(torch.full_like(q.data, 0.5))

def q_cosh(q: QuaternionTensor) -> QuaternionTensor:
    result = (q_exp(q) + q_exp(-q)) * QuaternionTensor(torch.full_like(q.data, 0.5))
    result.data = torch.clamp(result.data, -1e3, 1e3)
    return result

def coord_to_quaternion(x: torch.Tensor, y: torch.Tensor) -> QuaternionTensor:
    """Convert x,y coordinates to quaternion. x and y should be broadcastable tensors"""
    zeros = torch.zeros_like(x)
    return QuaternionTensor(torch.stack([zeros, x, y, zeros], dim=-1))

def q_add(x: QuaternionTensor, y: QuaternionTensor) -> QuaternionTensor:
    result = x + y
    result.data = torch.clamp(result.data, -1e6, 1e6)
    if result.data.shape != x.data.shape:
        result.data = result.data.view(x.data.shape)
    return result

def q_sub(x: QuaternionTensor, y: QuaternionTensor) -> QuaternionTensor:
    result = x - y
    result.data = torch.clamp(result.data, -1e6, 1e6)
    if result.data.shape != x.data.shape:
        result.data = result.data.view(x.data.shape)
    return result

def q_mul(x: QuaternionTensor, y: QuaternionTensor) -> QuaternionTensor:
    result = x * y
    result.data = torch.clamp(result.data, -1e6, 1e6)
    if result.data.shape != x.data.shape:
        result.data = result.data.view(x.data.shape)
    return result

def q_div(x: QuaternionTensor, y: QuaternionTensor) -> QuaternionTensor:
    # Add epsilon to avoid division by zero
    epsilon = 1e-10
    norm_squared = y.norm_squared().unsqueeze(-1)
    safe_denom = torch.where(norm_squared > epsilon, norm_squared, epsilon * torch.ones_like(norm_squared))
    result = x * QuaternionTensor(y.conjugate().data / safe_denom)
    result.data = torch.clamp(result.data, -1e6, 1e6)
    if result.data.shape != x.data.shape:
        result.data = result.data.view(x.data.shape)
    return result

def q_conjugate(q: QuaternionTensor) -> QuaternionTensor:
    result = q.conjugate()
    result.data = torch.clamp(result.data, -1e6, 1e6)
    if result.data.shape != q.data.shape:
        result.data = result.data.view(q.data.shape)
    return result

def q_normalize(q: QuaternionTensor) -> QuaternionTensor:
    epsilon = 1e-10
    norm = torch.sqrt(q.norm_squared()).unsqueeze(-1)
    safe_norm = torch.where(norm > epsilon, norm, epsilon * torch.ones_like(norm))
    result = QuaternionTensor(q.data / safe_norm)
    if result.data.shape != q.data.shape:
        result.data = result.data.view(q.data.shape)
    return result

def q_rotate45(q: QuaternionTensor) -> QuaternionTensor:
    result = q.rotate(torch.pi/4)
    result.data = torch.clamp(result.data, -1e6, 1e6)
    if result.data.shape != q.data.shape:
        result.data = result.data.view(q.data.shape)
    return result

def q_floor(q: QuaternionTensor) -> QuaternionTensor:
    result = q.floor()
    if result.data.shape != q.data.shape:
        result.data = result.data.view(q.data.shape)
    return result

def q_mod2(q: QuaternionTensor) -> QuaternionTensor:
    result = q.modulo(2.0)
    if result.data.shape != q.data.shape:
        result.data = result.data.view(q.data.shape)
    return result

def q_spiral(q: QuaternionTensor) -> QuaternionTensor:
    # Creates spiral patterns
    result = q * QuaternionTensor(torch.exp(q.data * 0.5))
    result.data = torch.clamp(result.data, -1e6, 1e6)
    if result.data.shape != q.data.shape:
        result.data = result.data.view(q.data.shape)
    return result

def q_wave(q: QuaternionTensor) -> QuaternionTensor:
    # Creates wave-like patterns
    result = q_sin(q) * q_cos(q.rotate(torch.pi/3))
    result.data = torch.clamp(result.data, -1e6, 1e6)
    if result.data.shape != q.data.shape:
        result.data = result.data.view(q.data.shape)
    return result

def q_blend(x: QuaternionTensor, y: QuaternionTensor) -> QuaternionTensor:
    # Smooth blend between quaternions
    t = 0.5 * (1 + torch.sin(x.norm_squared()))
    result = QuaternionTensor(t.unsqueeze(-1) * x.data + (1-t).unsqueeze(-1) * y.data)
    result.data = torch.clamp(result.data, -1e6, 1e6)
    if result.data.shape != x.data.shape:
        result.data = result.data.view(x.data.shape)
    return result

def q_ripple(q: QuaternionTensor) -> QuaternionTensor:
    # Creates ripple patterns
    r = torch.sqrt(q.x**2 + q.y**2)
    result = q_sin(QuaternionTensor(r.unsqueeze(-1) * torch.ones_like(q.data)))
    result.data = torch.clamp(result.data, -1e6, 1e6)
    if result.data.shape != q.data.shape:
        result.data = result.data.view(q.data.shape)
    return result

def q_swirl(q: QuaternionTensor) -> QuaternionTensor:
    # Creates swirl patterns
    r = torch.sqrt(q.x**2 + q.y**2)
    theta = torch.atan2(q.y, q.x) + r
    result = QuaternionTensor(torch.stack([
        torch.zeros_like(r),
        r * torch.cos(theta),
        r * torch.sin(theta),
        torch.zeros_like(r)
    ], dim=-1))
    if result.data.shape != q.data.shape:
        result.data = result.data.view(q.data.shape)
    return result

def q_ilog(q: QuaternionTensor) -> QuaternionTensor:
    # Logarithm with i-component emphasis
    result = q_log(Q_I * q)
    result.data = torch.clamp(result.data, -1e6, 1e6)
    if result.data.shape != q.data.shape:
        result.data = result.data.view(q.data.shape)
    return result

def q_isin(q: QuaternionTensor) -> QuaternionTensor:
    # Sine with i-component emphasis
    result = q_sin(Q_I * q)
    result.data = torch.clamp(result.data, -1e6, 1e6)
    if result.data.shape != q.data.shape:
        result.data = result.data.view(q.data.shape)
    return result

def q_iexp(q: QuaternionTensor) -> QuaternionTensor:
    # Q_I * q may be causing shape issues
    temp = Q_I * q
    result = q_exp(temp)
    if result.data.shape != q.data.shape:
        result.data = result.data.view(q.data.shape)
    result.data = torch.clamp(result.data, -1e6, 1e6)
    return result

def q_imin(x: QuaternionTensor, y: QuaternionTensor) -> QuaternionTensor:
    # The where operation might be breaking shapes
    norm_x = (Q_I * x).norm_squared().unsqueeze(-1)
    norm_y = (Q_I * y).norm_squared().unsqueeze(-1)
    result = QuaternionTensor(torch.where(norm_x < norm_y, x.data, y.data))
    if result.data.shape != x.data.shape:
        result.data = result.data.view(x.data.shape)
    return result

def q_imax(x: QuaternionTensor, y: QuaternionTensor) -> QuaternionTensor:
    # Similar shape fix for imax
    norm_x = (Q_I * x).norm_squared().unsqueeze(-1)
    norm_y = (Q_I * y).norm_squared().unsqueeze(-1) 
    result = QuaternionTensor(torch.where(norm_x > norm_y, x.data, y.data))
    if result.data.shape != x.data.shape:
        result.data = result.data.view(x.data.shape)
    return result

def q_rolR(q: QuaternionTensor) -> QuaternionTensor:
    # Roll operation might need shape preservation
    data = q.data.clone()
    result = QuaternionTensor(torch.roll(data, 1, dims=-1))
    if result.data.shape != q.data.shape:
        result.data = result.data.view(q.data.shape)
    return result

# Constants as QuaternionTensors
Q_IDENTITY = QuaternionTensor(torch.tensor([1., 0., 0., 0.]))
Q_I = QuaternionTensor(torch.tensor([0., 1., 0., 0.]))
Q_J = QuaternionTensor(torch.tensor([0., 0., 1., 0.]))
Q_K = QuaternionTensor(torch.tensor([0., 0., 0., 1.]))
Q_ZERO = QuaternionTensor(torch.tensor([0., 0., 0., 0.]))

# Constants list
CONSTANTS = [
    (lambda: Q_IDENTITY, 'CONSTANT'),
    (lambda: Q_I, 'CONSTANT'),
    (lambda: Q_J, 'CONSTANT'),
    (lambda: Q_K, 'CONSTANT'),
    (lambda: QuaternionTensor(torch.tensor([GOLDEN_RATIO, 0., 0., 0.])), 'CONSTANT')
]

# Coordinates list 
COORDINATES = [
    (coord_to_quaternion, 'COORDINATE'),
    (lambda x,y: QuaternionTensor(torch.stack([torch.zeros_like(x), x, torch.zeros_like(x), torch.zeros_like(x)], dim=-1)), 'COORDINATE'),
    (lambda x,y: QuaternionTensor(torch.stack([torch.zeros_like(y), torch.zeros_like(y), y, torch.zeros_like(y)], dim=-1)), 'COORDINATE'),
    (lambda x,y: QuaternionTensor(torch.stack([torch.zeros_like(x), y, x, torch.zeros_like(x)], dim=-1)), 'COORDINATE'),
    (lambda x,y: QuaternionTensor(torch.stack([x*y, x, y, torch.zeros_like(x)], dim=-1)), 'COORDINATE'),
    (lambda x,y: QuaternionTensor(torch.stack([x*x - y*y, x, y, torch.zeros_like(x)], dim=-1)), 'COORDINATE'),
    (lambda x,y: QuaternionTensor(torch.stack([torch.sin(x)*torch.cos(y), x, y, torch.zeros_like(x)], dim=-1)), 'COORDINATE')
]


# Update the OPERATORS list
OPERATORS = [
    (q_add, 'BINARY'),
    (q_sub, 'BINARY'), 
    (q_mul, 'BINARY'),
    (q_div, 'BINARY'),
    (q_inverse, 'UNARY'),
    (q_cube, 'UNARY'),
    (q_sinh, 'UNARY'),
    (q_cosh, 'UNARY'),
    (q_conjugate, 'UNARY'),
    (q_normalize, 'UNARY'),
    (q_rotate45, 'UNARY'),
    (q_floor, 'UNARY'),
    (q_mod2, 'UNARY'),
    (q_sin, 'UNARY'),
    (q_cos, 'UNARY'),
    (q_tan, 'UNARY'),
    (q_exp, 'UNARY'),
    (q_log, 'UNARY'),
    (q_abs, 'UNARY'),
    (q_sqrt, 'UNARY'),
    (q_power, 'UNARY'),
    (q_spiral, 'UNARY'),
    (q_wave, 'UNARY'),
    (q_blend, 'BINARY'),
    (q_ripple, 'UNARY'),
    (q_swirl, 'UNARY'),
    (q_iexp, 'UNARY'),
    (q_ilog, 'UNARY'),  
    (q_isin, 'UNARY'),
    (q_imin, 'BINARY'),
    (q_imax, 'BINARY'),
    (q_rolR, 'UNARY'),
    (coord_to_quaternion, 'COORDINATE'),
]

# Update operator mappings
OPERATOR_TO_STRING = {
    q_add: '+',
    q_sub: '-', 
    q_mul: '*',
    q_div: '/',
    q_inverse: 'inv',
    q_cube: 'cube',
    q_sinh: 'sinh',
    q_cosh: 'cosh',
    q_sin: 'sin',
    q_cos: 'cos',
    q_tan: 'tan',
    q_exp: 'exp',
    q_log: 'log',
    q_abs: 'abs',
    q_sqrt: 'sqrt',
    q_power: 'pow',
    q_conjugate: 'conj',
    q_normalize: 'norm',
    q_rotate45: 'rot45',
    q_floor: 'floor', 
    q_mod2: 'mod2',
    q_spiral: 'spiral',
    q_wave: 'wave',
    q_blend: 'blend',
    q_ripple: 'ripple',
    q_swirl: 'swirl',
    q_iexp: 'iexp',
    q_ilog: 'ilog',
    q_isin: 'isin', 
    q_imin: 'imin',
    q_imax: 'imax',
    q_rolR: 'rolR',
    coord_to_quaternion: 'coord',
}

STRING_TO_OPERATOR = {v: k for k, v in OPERATOR_TO_STRING.items()}

class ExpressionNode:
    def __init__(self, 
                 operator: Optional[Tuple[Callable, str]] = None,
                 left: Optional['ExpressionNode'] = None,
                 right: Optional['ExpressionNode'] = None):
        self.operator = operator[0] if operator else None
        self.op_type = operator[1] if operator else None
        self.left = left
        self.right = right
        self.value: Optional[QuaternionTensor] = None
        self.constant = False
        
    def evaluate(self, coords: QuaternionTensor) -> QuaternionTensor:
        try:
            #print(f"Evaluating: Operator={self.operator}, OpType={self.op_type}")
            if self.constant and self.value is not None:
                result = QuaternionTensor(self.value.data.expand_as(coords.data))
                #print(f"Constant value shape: {result.data.shape}")
                return result

            if self.operator is None:
                #print(f"Return coordinates shape: {coords.data.shape}")
                return coords

            if self.op_type == 'COORDINATE':
                result = self.operator(coords.x, coords.y)
            elif self.op_type == 'CONSTANT':
                if not self.value:
                    self.value = self.operator()
                    self.constant = True
                result = QuaternionTensor(self.value.data.expand_as(coords.data))
            else:
                left_val = self.left.evaluate(coords) if self.left else coords
                #print(f"Left value shape: {left_val.data.shape}; operator=", self.operator)
                if self.op_type == 'UNARY':
                    result = self.operator(left_val)
                else:
                    right_val = self.right.evaluate(coords) if self.right else coords
                    #print(f"Right value shape: {right_val.data.shape}; operator=", self.operator)
                    result = self.operator(left_val, right_val)
                    # Ensure consistent shape
            if result.data.shape != coords.data.shape:
                #print(f"Shape mismatch detected: {result.data.shape} resetting to {coords.data.shape}")
                result.data = result.data.reshape(coords.data.shape)

            #print(f"Result shape: {result.data.shape}, dtype: {result.data.dtype}")
            return result

        except Exception as e:
            logger.error(f"Error in evaluation: {e}")
            raise


    def mutate(self, rate: float = 0.1) -> None:
        """Randomly mutate the expression tree"""
        if random.random() < rate:
            # Only select operators with string mappings
            valid_ops = [(op, op_type) for op, op_type in OPERATORS 
                        if op in OPERATOR_TO_STRING]
            new_op = random.choice(valid_ops)
            self.operator = new_op[0]
            self.op_type = new_op[1]
            self.constant = False
            self.value = None

        if self.left:
            self.left.mutate(rate)
        if self.right:
            self.right.mutate(rate)

    def breed(self, other: 'ExpressionNode') -> 'ExpressionNode':
        """Create new expression by combining with another"""
        if random.random() < 0.5:
            new_expr = self._copy()
            donor = other
        else:
            new_expr = other._copy()
            donor = self

        target_node = new_expr._random_node()
        donor_node = donor._random_node()

        if target_node.left:
            target_node.left = donor_node._copy()
        else:
            target_node.right = donor_node._copy()

        return new_expr

    def _copy(self) -> 'ExpressionNode':
        """Create deep copy of expression tree"""
        new_node = ExpressionNode(
            (self.operator, self.op_type) if self.operator else None
        )
        if self.left:
            new_node.left = self.left._copy()
        if self.right:
            new_node.right = self.right._copy()
        if self.constant and self.value is not None:
            new_node.value = self.value
            new_node.constant = True
        return new_node

    def _random_node(self) -> 'ExpressionNode':
        """Get random node from expression tree"""
        nodes = []
        def collect(node):
            nodes.append(node)
            if node.left:
                collect(node.left)
            if node.right:
                collect(node.right)
        collect(self)
        return random.choice(nodes)

    @classmethod
    def create_random(cls, depth: int = 3) -> 'ExpressionNode':
        """Create random expression tree"""
        if depth <= 0:
            return cls((coord_to_quaternion, 'COORDINATE'))

        valid_ops = [(op, op_type) for op, op_type in OPERATORS 
                    if op in OPERATOR_TO_STRING]
        op = random.choice(valid_ops)
        node = cls(op)

        if op[1] in ['UNARY', 'BINARY']:
            node.left = cls.create_random(depth - 1)
            if op[1] == 'BINARY':
                node.right = cls.create_random(depth - 1)

        return node

    def to_string(self) -> str:
        """Convert expression tree to prefix notation string"""
        if self.operator is None:
            return "coord"

        op_str = OPERATOR_TO_STRING.get(self.operator, 'unknown')

        if self.op_type == 'COORDINATE':
            return f"({op_str})"

        if self.op_type == 'UNARY':
            left_str = self.left.to_string() if self.left else "coord"
            return f"({op_str} {left_str})"

        if self.op_type == 'BINARY':
            left_str = self.left.to_string() if self.left else "coord"
            right_str = self.right.to_string() if self.right else "coord"
            return f"({op_str} {left_str} {right_str})"

        return "coord"

    @classmethod
    def from_string(cls, formula: str) -> 'ExpressionNode':
        """Parse prefix notation string into expression tree"""
        def parse(tokens: list, index: int) -> tuple['ExpressionNode', int]:
            if index >= len(tokens):
                return cls((coord_to_quaternion, 'COORDINATE')), index

            token = tokens[index]

            if token == '(':
                # Read operator
                op_token = tokens[index + 1]
                if op_token not in STRING_TO_OPERATOR:
                    raise ValueError(f"Unknown operator: {op_token}")

                operator = STRING_TO_OPERATOR[op_token]
                # Find operator type from OPERATORS list
                op_type = None
                for op, type_ in OPERATORS:
                    if op == operator:
                        op_type = type_
                        break

                if op_type is None:
                    raise ValueError(f"Could not determine operator type for: {op_token}")

                node = cls((operator, op_type))

                if op_type == 'COORDINATE':
                    return node, index + 3  # Skip past closing paren

                # Parse left child
                node.left, index = parse(tokens, index + 2)

                if op_type == 'BINARY':
                    # Parse right child for binary operators
                    node.right, index = parse(tokens, index)

                # Skip past closing paren
                return node, index + 1

            elif token == ')':
                return cls((coord_to_quaternion, 'COORDINATE')), index + 1
            else:
                # Handle bare coordinate
                return cls((coord_to_quaternion, 'COORDINATE')), index + 1

        # Split into tokens preserving parentheses 
        tokens = []
        current = ''
        for char in formula:
            if char in '()':
                if current:
                    tokens.append(current)
                    current = ''
                tokens.append(char)
            elif char.isspace():
                if current:
                    tokens.append(current)
                    current = ''
            else:
                current += char
        if current:
            tokens.append(current)

        node, _ = parse(tokens, 0)
        return node

    def __str__(self) -> str:
        return self.to_string()

class ImageGenerator:
    def __init__(self, width: int = 64, height: int = 64, device: Optional[torch.device] = None):
        self.width = width 
        self.height = height
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Pre-compute coordinate grid
        x_center = width / 2
        y_center = height / 2
        scale = min(width, height) / 4
        
        x = torch.linspace(0, width-1, width, device=self.device)
        y = torch.linspace(0, height-1, height, device=self.device)
        Y, X = torch.meshgrid(y, x, indexing='ij')
        
        X = (X - x_center) / scale
        Y = (Y - y_center) / scale
        
        # Create coordinate quaternions
        coords = torch.stack([
            torch.zeros_like(X),
            X, 
            Y,
            torch.zeros_like(X)
        ], dim=-1)
        
        self.coords = QuaternionTensor(coords)

    @torch.no_grad()
    def generate(self, expression: ExpressionNode) -> Image.Image:
        try:
            result = expression.evaluate(self.coords)
            #print(f"Generated result shape: {result.data.shape}, dtype: {result.data.dtype}")
            rgb_values = result.to_rgb()

            result_array = rgb_values.cpu().numpy()
            #print(f"Final image array shape: {result_array.shape}, dtype: {result_array.dtype}")

            return Image.fromarray(result_array.astype(np.uint8))
        except Exception as e:
            logger.error(f"Error generating image: {e}")
            return Image.fromarray(
                np.zeros((self.height, self.width, 3), dtype=np.uint8)
            )


def test_pytorch_main():
    # Set random seed for reproducibility
    # torch.manual_seed(2)
    # random.seed(2)
    
    # Select device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create timestamp for output folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"pytorch_images_{timestamp}"
    os.makedirs(folder_name, exist_ok=True)
    
    image_size = 64
    generator = ImageGenerator(image_size, image_size, device=device)
    
    # Generate multiple images
    num_images = 1000
    image_formulas = {}
    
    print(f"\nGenerating {num_images} images...")
    start_time = time.time()
    
    for i in range(num_images):
        try:
            # Clear GPU memory before each generation
            torch.cuda.empty_cache()
            
            # Create random expression with reduced depth
            expr = ExpressionNode.create_random(depth=7)  # Reduced depth
            formula_str = expr.to_string()
            image_formulas[f"image_{i}"] = formula_str
            #print(f"Image {i+1}: {formula_str}")
            
            # Generate and save image
            img = generator.generate(expr)
            image_path = os.path.join(folder_name, f"image_{i}.png")
            img.save(image_path)
            
            #print(f"Generated image {i+1}/{num_images}")
            
        except Exception as e:
            logger.error(f"Error generating image: {e}")
            continue
    
    end_time = time.time()
    total_time = end_time - start_time
    avg_time = total_time / num_images
    
    print(f"\nGeneration complete!")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Average time per image: {avg_time:.5f} seconds")
    print(f"\nFiles saved in '{folder_name}':")
    print(f"- {num_images} images (image_0.png to image_{num_images-1}.png)")

if __name__ == "__main__":
    test_pytorch_main()