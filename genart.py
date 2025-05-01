import logging
import math
import random
import torch
import numpy as np
from dataclasses import dataclass
from typing import Optional
from enum import Enum, auto
from PIL import Image
import torchvision.transforms
import os
import time

from timing_utils import time_it

# Conditionally import log_event if the module exists
try:
    from logging_utils import log_event
    HAS_LOG_EVENT = True
except ImportError:
    HAS_LOG_EVENT = False
    # Define a dummy log_event function that does nothing
    def log_event(step, event_type, agent_id, details=None):
        pass

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class OpType(Enum):
    CONSTANT = auto()
    UNARY = auto()
    BINARY = auto()
    COORDINATE = auto()

@dataclass
class QuaternionTensor:
    """
    A dataclass that encapsulates a PyTorch tensor representing a quaternion [w, x, y, z].
    """
    data: torch.Tensor

    @property
    def device(self):
        return self.data.device

    @property
    def w(self):
        return self.data[..., 0]

    @property
    def x(self):
        return self.data[..., 1]

    @property
    def y(self):
        return self.data[..., 2]

    @property
    def z(self):
        return self.data[..., 3]

    def __neg__(self) -> 'QuaternionTensor':
        """
        Implement unary minus for Quaternions (fixes error with expressions like -q).
        """
        return QuaternionTensor(-self.data)

    def __add__(self, other: 'QuaternionTensor') -> 'QuaternionTensor':
        return QuaternionTensor(self.data + other.data)

    def __sub__(self, other: 'QuaternionTensor') -> 'QuaternionTensor':
        return QuaternionTensor(self.data - other.data)

    def __mul__(self, other: 'QuaternionTensor') -> 'QuaternionTensor':
        """
        Quaternion multiplication.
        """
        w1, x1, y1, z1 = self.w, self.x, self.y, self.z
        w2, x2, y2, z2 = other.w, other.x, other.y, other.z
        return QuaternionTensor(torch.stack([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ], dim=-1))
    
    @staticmethod
    def batch_mul(q1_data, q2_data):
        """
        Optimized quaternion multiplication for batches.
        Takes raw tensor data instead of QuaternionTensor objects.
        """
        w1, x1, y1, z1 = q1_data[..., 0], q1_data[..., 1], q1_data[..., 2], q1_data[..., 3]
        w2, x2, y2, z2 = q2_data[..., 0], q2_data[..., 1], q2_data[..., 2], q2_data[..., 3]
        
        return torch.stack([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ], dim=-1)

    def __truediv__(self, other: 'QuaternionTensor') -> 'QuaternionTensor':
        """
        Quaternion division by multiplying with inverse of 'other'.
        """
        denom = other.norm_squared().unsqueeze(-1)
        mask = denom > 1e-10
        recip = other.conjugate()
        safe_denom = torch.where(mask, denom, torch.ones_like(denom))
        result = self * QuaternionTensor(recip.data / safe_denom)
        result.data = torch.where(mask.unsqueeze(-1), result.data, torch.zeros_like(result.data))
        return result

    def rotate(self, angle: float) -> 'QuaternionTensor':
        """
        Rotate quaternion by a real angle around the x-axis, for example.
        """
        c = torch.cos(torch.tensor(angle / 2.0, device=self.device))
        s = torch.sin(torch.tensor(angle / 2.0, device=self.device))
        return QuaternionTensor(torch.stack([
            c*self.w - s*self.x,
            c*self.x + s*self.w,
            c*self.y - s*self.z,
            c*self.z + s*self.y
        ], dim=-1))
    
    @staticmethod
    def batch_rotate(data, angle):
        """
        Apply rotation to batch of quaternion data
        """
        c = torch.cos(torch.tensor(angle / 2.0, device=data.device))
        s = torch.sin(torch.tensor(angle / 2.0, device=data.device))
        w, x, y, z = data[..., 0], data[..., 1], data[..., 2], data[..., 3]
        return torch.stack([
            c*w - s*x,
            c*x + s*w,
            c*y - s*z,
            c*z + s*y
        ], dim=-1)

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
        
    @staticmethod
    def batch_conjugate(data):
        """
        Optimized batch conjugate operation
        """
        return torch.stack([
            data[..., 0],
            -data[..., 1], 
            -data[..., 2], 
            -data[..., 3]
        ], dim=-1)

    def norm_squared(self) -> torch.Tensor:
        return torch.sum(self.data * self.data, dim=-1)

    def normalize(self) -> 'QuaternionTensor':
        """
        Normalize the quaternion in place.
        """
        norm = torch.sqrt(self.norm_squared()).unsqueeze(-1)
        mask = norm > 1e-10
        safe_norm = torch.where(mask, norm, torch.ones_like(norm))
        out = QuaternionTensor(self.data / safe_norm)
        out.data = torch.where(mask, out.data, torch.zeros_like(out.data))
        return out

    def to_rgb(self) -> torch.Tensor:
        """
        Convert last three channels [x, y, z] to an RGB pixel in [0..255].
        """
        if self.data.shape[-1] != 4:
            raise ValueError(f"Quaternion tensor must have 4 components, got {self.data.shape[-1]}")
        values = torch.clamp(self.data[..., 1:4], -10.0, 10.0)
        rgb = 255.0 / (1.0 + torch.exp(-values))
        if len(rgb.shape) < 3 or rgb.shape[-1] != 3:
            raise ValueError(f"RGB tensor must have 3 channels, got shape {rgb.shape}")
        return rgb.to(torch.uint8)
        
    @staticmethod
    def batch_to_rgb(data):
        """
        Convert batch of quaternion data to RGB
        """
        values = torch.clamp(data[..., 1:4], -10.0, 10.0)
        rgb = 255.0 / (1.0 + torch.exp(-values))
        return rgb.to(torch.uint8)


# Common quaternion constants
Q_IDENTITY = QuaternionTensor(torch.tensor([1., 0., 0., 0.]))
Q_I = QuaternionTensor(torch.tensor([0., 1., 0., 0.]))
Q_J = QuaternionTensor(torch.tensor([0., 0., 1., 0.]))
Q_K = QuaternionTensor(torch.tensor([0., 0., 0., 1.]))
GOLDEN_RATIO = (1 + torch.sqrt(torch.tensor(5.0))) / 2
Q_ZERO = QuaternionTensor(torch.tensor([0., 0., 0., 0.]))

def coord_to_quaternion(x: torch.Tensor, y: torch.Tensor) -> QuaternionTensor:
    """
    Convert 2D coordinates to a quaternion [0, x, y, 0].
    """
    zeros = torch.zeros_like(x)
    return QuaternionTensor(torch.stack([zeros, x, y, zeros], dim=-1))


def q_sin(q: QuaternionTensor) -> QuaternionTensor:
    return QuaternionTensor(torch.sin(q.data))

def q_cos(q: QuaternionTensor) -> QuaternionTensor:
    return QuaternionTensor(torch.cos(q.data))

def q_tan(q: QuaternionTensor) -> QuaternionTensor:
    return q_sin(q) * q_cos(q.rotate(math.pi / 3.0)).conjugate()

def q_exp(q: QuaternionTensor) -> QuaternionTensor:
    return QuaternionTensor(torch.exp(torch.clamp(q.data, -30.0, 30.0)))

def q_log(q: QuaternionTensor) -> QuaternionTensor:
    safe_data = torch.where(q.data > 1e-10, q.data, torch.tensor(1e-10, device=q.device))
    return QuaternionTensor(torch.log(safe_data))

def q_sqrt(q: QuaternionTensor) -> QuaternionTensor:
    safe_data = torch.where(q.data >= 0, q.data, torch.tensor(0.0, device=q.device))
    return QuaternionTensor(torch.sqrt(safe_data))

def q_abs(q: QuaternionTensor) -> QuaternionTensor:
    abs_data = torch.abs(q.data)
    return QuaternionTensor(abs_data.reshape(q.data.shape))

def q_power(q: QuaternionTensor) -> QuaternionTensor:
    """
    Raise quaternion to a random real power p in [0,1].
    """
    p = torch.rand(1).item()
    p_tensor = torch.tensor(p, device=q.device)
    return q_exp(q_log(q) * QuaternionTensor(torch.full_like(q.data, p_tensor)))

def q_inverse(q: QuaternionTensor) -> QuaternionTensor:
    norm_squared = q.norm_squared().unsqueeze(-1)
    mask = norm_squared > 1e-10
    recip = q.conjugate()
    safe_norm = torch.where(mask, norm_squared, torch.ones_like(norm_squared))
    result = QuaternionTensor(recip.data / safe_norm)
    if result.data.ndim > 3:
        result.data = result.data.squeeze()
    return result

def q_cube(q: QuaternionTensor) -> QuaternionTensor:
    result = q * q * q
    result.data = torch.clamp(result.data, -1e3, 1e3)
    if result.data.ndim > 3:
        result.data = result.data.squeeze()
    return result

def q_sinh(q: QuaternionTensor) -> QuaternionTensor:
    return (q_exp(q) - q_exp(-q)) * QuaternionTensor(torch.full_like(q.data, 0.5))

def q_cosh(q: QuaternionTensor) -> QuaternionTensor:
    result = (q_exp(q) + q_exp(-q)) * QuaternionTensor(torch.full_like(q.data, 0.5))
    result.data = torch.clamp(result.data, -1e3, 1e3)
    return result

def q_conjugate(q: QuaternionTensor) -> QuaternionTensor:
    return q.conjugate()

def q_normalize(q: QuaternionTensor) -> QuaternionTensor:
    """
    L2-normalize the quaternion's 4D vector.
    """
    epsilon = 1e-10
    norm = torch.sqrt(q.norm_squared()).unsqueeze(-1)
    safe_norm = torch.where(norm > epsilon, norm, epsilon * torch.ones_like(norm))
    data = q.data / safe_norm
    data = torch.clamp(data, -1e6, 1e6)
    return QuaternionTensor(data)

def q_rotate45(q: QuaternionTensor) -> QuaternionTensor:
    result = q.rotate(math.pi / 4.0)
    result.data = torch.clamp(result.data, -1e6, 1e6)
    return result

def q_floor(q: QuaternionTensor) -> QuaternionTensor:
    result = q.floor()
    return result

def q_mod2(q: QuaternionTensor) -> QuaternionTensor:
    result = q.modulo(2.0)
    return result

def q_spiral(q: QuaternionTensor) -> QuaternionTensor:
    """
    Creates spiral-like patterns.
    """
    result = q * QuaternionTensor(torch.exp(q.data * 0.5))
    result.data = torch.clamp(result.data, -1e6, 1e6)
    return result

def q_wave(q: QuaternionTensor) -> QuaternionTensor:
    """
    Creates wave-like patterns with sin/cos.
    """
    result = q_sin(q) * q_cos(q.rotate(math.pi / 3.0))
    result.data = torch.clamp(result.data, -1e6, 1e6)
    return result

def q_blend(x: QuaternionTensor, y: QuaternionTensor) -> QuaternionTensor:
    """
    Blends two quaternions in a wave-based manner.
    """
    t = 0.5 * (1 + torch.sin(x.norm_squared()))
    data = t.unsqueeze(-1) * x.data + (1 - t).unsqueeze(-1) * y.data
    data = torch.clamp(data, -1e6, 1e6)
    return QuaternionTensor(data)

def q_ripple(q: QuaternionTensor) -> QuaternionTensor:
    r = torch.sqrt(q.x**2 + q.y**2)
    wave_input = QuaternionTensor(r.unsqueeze(-1).expand_as(q.data))
    result = q_sin(wave_input)
    result.data = torch.clamp(result.data, -1e6, 1e6)
    return result

def q_swirl(q: QuaternionTensor) -> QuaternionTensor:
    r = torch.sqrt(q.x**2 + q.y**2)
    theta = torch.atan2(q.y, q.x) + r
    result_data = torch.stack([
        torch.zeros_like(r),
        r * torch.cos(theta),
        r * torch.sin(theta),
        torch.zeros_like(r)
    ], dim=-1)
    return QuaternionTensor(result_data)

def q_iexp(q: QuaternionTensor) -> QuaternionTensor:
    """
    Exponential with an imaginary multiplier (Q_I).
    """
    temp = Q_I * q
    result = q_exp(temp)
    result.data = torch.clamp(result.data, -1e6, 1e6)
    return result

def q_ilog(q: QuaternionTensor) -> QuaternionTensor:
    """
    Logarithm with an imaginary multiplier.
    """
    result = q_log(Q_I * q)
    result.data = torch.clamp(result.data, -1e6, 1e6)
    return result

def q_isin(q: QuaternionTensor) -> QuaternionTensor:
    """
    Sine with an imaginary multiplier.
    """
    result = q_sin(Q_I * q)
    result.data = torch.clamp(result.data, -1e6, 1e6)
    return result

def q_imin(x: QuaternionTensor, y: QuaternionTensor) -> QuaternionTensor:
    """
    Compare norms after multiplying by Q_I, then choose minimum.
    """
    nx = (Q_I * x).norm_squared().unsqueeze(-1)
    ny = (Q_I * y).norm_squared().unsqueeze(-1)
    data = torch.where(nx < ny, x.data, y.data)
    return QuaternionTensor(data)

def q_imax(x: QuaternionTensor, y: QuaternionTensor) -> QuaternionTensor:
    """
    Compare norms after multiplying by Q_I, then choose maximum.
    """
    nx = (Q_I * x).norm_squared().unsqueeze(-1)
    ny = (Q_I * y).norm_squared().unsqueeze(-1)
    data = torch.where(nx > ny, x.data, y.data)
    return QuaternionTensor(data)

def q_rolR(q: QuaternionTensor) -> QuaternionTensor:
    data = q.data.clone()
    result = torch.roll(data, 1, dims=-1)
    return QuaternionTensor(result)

def q_add(x: QuaternionTensor, y: QuaternionTensor) -> QuaternionTensor:
    return x + y

def q_sub(x: QuaternionTensor, y: QuaternionTensor) -> QuaternionTensor:
    return x - y

def q_mul(x: QuaternionTensor, y: QuaternionTensor) -> QuaternionTensor:
    return x * y

def q_div(x: QuaternionTensor, y: QuaternionTensor) -> QuaternionTensor:
    return x / y

# Operators list (no 'coord_to_quaternion' in deeper expressions):
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
]

OPERATOR_TO_STRING = {
    q_add: '+',
    q_sub: '-',
    q_mul: '*',
    q_div: '/',
    q_inverse: 'inv',
    q_cube: 'cube',
    q_sinh: 'sinh',
    q_cosh: 'cosh',
    q_conjugate: 'conj',
    q_normalize: 'norm',
    q_rotate45: 'rot45',
    q_floor: 'floor',
    q_mod2: 'mod2',
    q_sin: 'sin',
    q_cos: 'cos',
    q_tan: 'tan',
    q_exp: 'exp',
    q_log: 'log',
    q_abs: 'abs',
    q_sqrt: 'sqrt',
    q_power: 'pow',
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
}
STRING_TO_OPERATOR = {v: k for k, v in OPERATOR_TO_STRING.items()}


def _force_shape(q: QuaternionTensor, target_shape: torch.Size) -> QuaternionTensor:
    """
    Force the quaternion data in 'q' to match 'target_shape' if possible.
    If the total number of elements differs, replaces with zeros.
    """
    if q.data.shape == target_shape:
        return q
    try:
        # Handle case where we have an extra dimension
        if len(q.data.shape) == 4 and len(target_shape) == 3:
            if q.data.shape[0] == target_shape[0]:
                # Take first slice if shapes match except for extra dimension
                q.data = q.data[..., 0, :]
                return q
            
        if torch.numel(q.data) == target_shape.numel():
            # Safe to reshape
            q.data = q.data.reshape(target_shape)
        else:
            logger.error(f"Cannot safely reshape from {q.data.shape} to {target_shape}; zeroing out.")
            q.data = torch.zeros(target_shape, device=q.data.device)
    except Exception as e:
        logger.error(f"Shape mismatch: {q.data.shape} vs {target_shape} -- zeroing out.")
        q.data = torch.zeros(target_shape, device=q.data.device)
    return q


class ExpressionNode:
    """
    Represents a node in the generative expression tree. 
    - If operator is None, we treat the node as just "coords".
    """

    def __init__(self, operator=None, left=None, right=None):
        self.operator = operator[0] if operator else None
        self.op_type = operator[1] if operator else None
        self.left = left
        self.right = right
        self.value = None
        self.constant = False
        self.eval_cache = {}  # Cache for evaluation results

    @time_it
    def evaluate(self, coords: QuaternionTensor) -> QuaternionTensor:
        """
        Evaluate this expression node, returning a QuaternionTensor
        guaranteed to have the same shape as coords.
        """
        device = coords.device
        coords_shape = coords.data.shape

        # Use caching for inputs with the same shape
        cache_key = (coords_shape, device.type)
        if self.constant and cache_key in self.eval_cache:
            return self.eval_cache[cache_key]

        # If we have a cached constant, just expand to coords shape
        if self.constant and self.value is not None:
            q_out = QuaternionTensor(self.value.data.to(device))
            result = _force_shape(q_out, coords_shape)
            self.eval_cache[cache_key] = result
            return result

        # No operator => "leaf" node => just coords
        if self.operator is None:
            return coords  # No need to force shape for leaf nodes

        try:
            if self.op_type == 'UNARY':
                left_val = self.left.evaluate(coords) if self.left else coords
                if left_val.data.shape != coords_shape:
                    left_val = _force_shape(left_val, coords_shape)
                out = self.operator(left_val)

            elif self.op_type == 'BINARY':
                left_val = self.left.evaluate(coords) if self.left else coords
                if left_val.data.shape != coords_shape:
                    left_val = _force_shape(left_val, coords_shape)
                    
                right_val = self.right.evaluate(coords) if self.right else coords
                if right_val.data.shape != coords_shape:
                    right_val = _force_shape(right_val, coords_shape)
                    
                out = self.operator(left_val, right_val)

            else:
                # If we ever get here, fallback to coords
                return coords

            # Only force shape if necessary
            if out.data.shape != coords_shape:
                out = _force_shape(out, coords_shape)
                
            return out

        except Exception as e:
            logger.error(f"Error in ExpressionNode.evaluate: {e}")
            # Fallback to zero on error
            return QuaternionTensor(torch.zeros(coords_shape, device=device))

    def mutate(self, rate=0.1, agent_id=None, step=None):
        """
        Randomly mutate this node's operator, and recursively mutate children.
        
        Parameters
        ----------
        rate : float
            Probability of mutation at each node
        agent_id : int, optional
            If provided along with step, logs mutation events
        step : int, optional
            Current simulation step for logging
        """
        mutated_here = False
        original_op_str = OPERATOR_TO_STRING.get(self.operator, 'coord')
        
        if random.random() < rate:
            valid_ops = [(op, op_type) for op, op_type in OPERATORS
                         if op in OPERATOR_TO_STRING]
            new_op = random.choice(valid_ops)
            self.operator = new_op[0]
            self.op_type = new_op[1]
            self.constant = False
            self.value = None
            mutated_here = True

            # Log mutation if context provided and logging is available
            if HAS_LOG_EVENT and agent_id is not None and step is not None:
                log_event(
                    step=step,
                    event_type='mutation_applied',
                    agent_id=agent_id,
                    details={
                        'original_operator': original_op_str,
                        'new_operator': OPERATOR_TO_STRING.get(self.operator, 'unknown'),
                        # Consider adding node depth or path if needed
                    }
                )
                logger.debug(f"Agent {agent_id}: Mutated operator from {original_op_str} to {OPERATOR_TO_STRING.get(self.operator, 'unknown')}")

        if self.left:
            self.left.mutate(rate, agent_id=agent_id, step=step)
        if self.right:
            self.right.mutate(rate, agent_id=agent_id, step=step)
            
        return self  # Return self for chaining

    def breed(self, other: 'ExpressionNode', agent_id=None, step=None) -> 'ExpressionNode':
        """
        Cross-over / combine this node with another ExpressionNode.
        Respects MAX_EXPRESSION_DEPTH from config if available.
        
        Parameters
        ----------
        other : ExpressionNode
            The other parent expression to breed with
        agent_id : int, optional
            If provided along with step, logs breeding events
        step : int, optional
            Current simulation step for logging
            
        Returns
        -------
        ExpressionNode
            A new expression created by combining this one with other
        """
        import config
        
        # Decide which expression is base and which is donor
        if random.random() < 0.5:
            new_expr = self._copy()
            donor = other
            parent1_str = self.to_string() if HAS_LOG_EVENT else None  # Only compute if needed
            parent2_str = other.to_string() if HAS_LOG_EVENT else None
        else:
            new_expr = other._copy()
            donor = self
            parent1_str = other.to_string() if HAS_LOG_EVENT else None
            parent2_str = self.to_string() if HAS_LOG_EVENT else None

        # Select random nodes for crossover
        target_node = new_expr._random_node()
        donor_node = donor._random_node()
        donor_copy = donor_node._copy()

        # Perform crossover
        original_subtree_str = None
        if target_node.left and random.random() < 0.5:  # Replace left more often if exists
            if HAS_LOG_EVENT:
                original_subtree_str = target_node.left.to_string() if target_node.left else 'coord'
            target_node.left = donor_copy
        elif target_node.right:
            if HAS_LOG_EVENT:
                original_subtree_str = target_node.right.to_string() if target_node.right else 'coord'
            target_node.right = donor_copy
        else:  # If target is a leaf or unary with only left, replace left
            if HAS_LOG_EVENT:
                original_subtree_str = target_node.left.to_string() if target_node.left else 'coord'
            target_node.left = donor_copy

        # Limit depth if needed
        simplified = False
        if hasattr(config, 'MAX_EXPRESSION_DEPTH'):
            depth = new_expr._get_max_depth()
            if depth > config.MAX_EXPRESSION_DEPTH:
                simplified = new_expr._simplify_to_depth(config.MAX_EXPRESSION_DEPTH)
        
        # Log breed operation if context provided and logging is available
        if HAS_LOG_EVENT and agent_id is not None and step is not None:
            log_event(
                step=step,
                event_type='breed_operation',
                agent_id=agent_id,
                details={
                    'parent1_expression': parent1_str,
                    'parent2_expression': parent2_str,
                    'child_expression': new_expr.to_string(),
                    'donor_subtree': donor_node.to_string(),
                    'replaced_subtree': original_subtree_str,
                    'simplified_due_to_depth': simplified
                }
            )
            logger.debug(f"Agent {agent_id}: Bred new expression from parents")

        # Apply mutation after breeding sometimes
        if random.random() < 0.05:  # Lower mutation rate after breed
            new_expr.mutate(rate=0.05, agent_id=agent_id, step=step)
            
        return new_expr

    def _copy(self) -> 'ExpressionNode':
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
        """
        Return a random node from this expression tree (including self).
        """
        nodes = []

        def collect(n: ExpressionNode):
            nodes.append(n)
            if n.left:
                collect(n.left)
            if n.right:
                collect(n.right)

        collect(self)
        return random.choice(nodes)

    @classmethod
    def create_random(cls, depth=3):
        """
        Create a random expression tree of given depth.
        If depth <= 0 => treat node as just coords (None operator).
        Respects MAX_EXPRESSION_DEPTH from config if available.
        """
        import config
        
        # Limit maximum depth if configured
        if hasattr(config, 'MAX_EXPRESSION_DEPTH'):
            depth = min(depth, config.MAX_EXPRESSION_DEPTH)
            
        if depth <= 0:
            return cls(None)

        valid_ops = [(op, op_type) for op, op_type in OPERATORS
                     if op in OPERATOR_TO_STRING]
        op = random.choice(valid_ops)
        node = cls(op)
        if op[1] in ['UNARY', 'BINARY']:
            node.left = cls.create_random(depth - 1)
            if op[1] == 'BINARY':
                node.right = cls.create_random(depth - 1)
        return node

    def _get_max_depth(self) -> int:
        """
        Get the maximum depth of this expression tree.
        """
        if self.operator is None:
            return 0
        
        left_depth = self.left._get_max_depth() if self.left else 0
        right_depth = self.right._get_max_depth() if self.right else 0
        return 1 + max(left_depth, right_depth)

    def _simplify_to_depth(self, max_depth: int, current_depth: int = 0) -> bool:
        """
        Simplify the expression tree to respect max_depth.
        Returns True if the node was simplified.
        """
        if current_depth >= max_depth:
            # Replace this node with a simpler node
            self.operator = None
            self.op_type = None
            self.left = None
            self.right = None
            self.eval_cache = {}
            return True
            
        if self.left and self.left._simplify_to_depth(max_depth, current_depth + 1):
            self.left = None
            
        if self.right and self.right._simplify_to_depth(max_depth, current_depth + 1):
            self.right = None
            
        return False

    def clear_cache(self):
        """
        Clear the evaluation cache for this node and all children.
        """
        self.eval_cache = {}
        if self.left:
            self.left.clear_cache()
        if self.right:
            self.right.clear_cache()

    def to_string(self) -> str:
        """
        Convert expression tree to a prefix notation string.
        """
        if self.operator is None:
            return "(coord)"

        op_str = OPERATOR_TO_STRING.get(self.operator, 'unknown')
        if self.op_type == 'UNARY':
            left_str = self.left.to_string() if self.left else "(coord)"
            return f"({op_str} {left_str})"
        elif self.op_type == 'BINARY':
            left_str = self.left.to_string() if self.left else "(coord)"
            right_str = self.right.to_string() if self.right else "(coord)"
            return f"({op_str} {left_str} {right_str})"
        else:
            return "(coord)"

    @classmethod
    def from_string(cls, formula: str) -> 'ExpressionNode':
        """
        Parse prefix notation string to rebuild an ExpressionNode tree.
        """
        def parse(tokens, index):
            if index >= len(tokens):
                return cls(None), index

            token = tokens[index]
            if token == '(':
                # read operator or "coord"
                op_token = tokens[index + 1]
                if op_token == 'coord':
                    return cls(None), index + 2  # skip 'coord' + closing paren

                if op_token not in STRING_TO_OPERATOR:
                    raise ValueError(f"Unknown operator: {op_token}")

                operator = STRING_TO_OPERATOR[op_token]
                # find operator type
                op_type = None
                for (opf, t) in OPERATORS:
                    if opf == operator:
                        op_type = t
                        break

                node = cls((operator, op_type))
                if op_type == 'UNARY':
                    node.left, index = parse(tokens, index + 2)
                    return node, index + 1  # skip closing ')'
                elif op_type == 'BINARY':
                    node.left, index = parse(tokens, index + 2)
                    node.right, index = parse(tokens, index)
                    return node, index + 1
                else:
                    return cls(None), index + 1
            elif token == ')':
                return cls(None), index + 1
            else:
                # treat unknown token as coords
                return cls(None), index + 1

        tokens = []
        current = ''
        for c in formula:
            if c in '()':
                if current:
                    tokens.append(current)
                    current = ''
                tokens.append(c)
            elif c.isspace():
                if current:
                    tokens.append(current)
                    current = ''
            else:
                current += c
        if current:
            tokens.append(current)

        node, _ = parse(tokens, 0)
        return node

    def __str__(self) -> str:
        return self.to_string()


class ImageGenerator:
    """
    Utility class that pre-computes coordinate quaternions for a given width/height,
    then can generate images by evaluating ExpressionNodes.
    """

    def __init__(self, width=64, height=64, device=None):
        self.width = width
        self.height = height
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        x_center = width / 2.0
        y_center = height / 2.0
        scale = min(width, height) / 4.0

        x = torch.linspace(0, width - 1, width, device=self.device)
        y = torch.linspace(0, height - 1, height, device=self.device)
        Y, X = torch.meshgrid(y, x, indexing='ij')
        X = (X - x_center) / scale
        Y = (Y - y_center) / scale

        coords = torch.stack([
            torch.zeros_like(X),   # w = 0
            X,                     # x
            Y,                     # y
            torch.zeros_like(X)    # z = 0
        ], dim=-1)

        self.coords = QuaternionTensor(coords)

    @torch.no_grad()
    @time_it
    def generate(self, expression: ExpressionNode) -> Image.Image:
        """
        Evaluate the expression over the precomputed coordinate grid,
        convert to RGB, and produce a PIL Image.

        Parameters
        ----------
        expression : ExpressionNode

        Returns
        -------
        PIL.Image
        """
        try:
            result = expression.evaluate(self.coords)
            rgb_values = result.to_rgb()
            result_array = rgb_values.cpu().numpy()
            return Image.fromarray(result_array.astype(np.uint8))
        except Exception as e:
            logger.error(f"Error generating image: {e}")
            return Image.fromarray(np.zeros((self.height, self.width, 3), dtype=np.uint8))

    @torch.no_grad()
    @time_it
    def batch_generate(self, expressions, max_batch_size=None):
        """
        Generate multiple images at once by batching expressions.
        This is significantly faster than generating one by one.
        
        Parameters
        ----------
        expressions : list of ExpressionNode
            List of expressions to evaluate
        max_batch_size : int, optional
            Maximum number of expressions to process at once
            
        Returns
        -------
        list of PIL.Image
            Generated images
        """
        import config
        
        # Use config value if not explicitly provided
        if max_batch_size is None and hasattr(config, 'BATCH_SIZE_GENERATION'):
            max_batch_size = config.BATCH_SIZE_GENERATION
        elif max_batch_size is None:
            max_batch_size = 32
            
        images = []
        
        # Process in batches to avoid OOM errors
        for i in range(0, len(expressions), max_batch_size):
            batch = expressions[i:i + max_batch_size]
            batch_size = len(batch)
            
            # Check if we can use vectorized computation for simple expressions
            if (batch_size > 1 and 
                all(expr.operator == batch[0].operator for expr in batch) and
                batch[0].op_type == 'UNARY'):
                
                # These operators are easily vectorized
                vectorizable_ops = [q_sin, q_cos, q_floor, q_exp, q_abs, 
                                   q_sqrt, q_normalize, q_conjugate]
                
                if batch[0].operator in vectorizable_ops:
                    try:
                        # Try fully vectorized approach
                        all_inputs = []
                        
                        # Prepare coordinate data once per batch 
                        coords_data = self.coords.data
                        
                        # Get left values for all expressions
                        for expr in batch:
                            if expr.left:
                                left_val = expr.left.evaluate(self.coords)
                                all_inputs.append(left_val.data)
                            else:
                                all_inputs.append(coords_data)
                        
                        # Stack and process in one go
                        stacked_inputs = torch.stack(all_inputs)
                        
                        # Apply appropriate operation to the whole batch
                        if batch[0].operator == q_sin:
                            results = torch.sin(stacked_inputs)
                        elif batch[0].operator == q_cos:
                            results = torch.cos(stacked_inputs)
                        elif batch[0].operator == q_floor:
                            results = torch.floor(stacked_inputs)
                        elif batch[0].operator == q_exp:
                            results = torch.exp(torch.clamp(stacked_inputs, -30.0, 30.0))
                        elif batch[0].operator == q_abs:
                            results = torch.abs(stacked_inputs)
                        elif batch[0].operator == q_sqrt:
                            results = torch.sqrt(torch.clamp(stacked_inputs, 0, None))
                        elif batch[0].operator == q_normalize:
                            norms = torch.sqrt(torch.sum(stacked_inputs * stacked_inputs, dim=-1, keepdim=True))
                            safe_norms = torch.clamp(norms, min=1e-10)
                            results = stacked_inputs / safe_norms
                        elif batch[0].operator == q_conjugate:
                            results = QuaternionTensor.batch_conjugate(stacked_inputs)
                        
                        # Convert results to RGB
                        rgb_tensors = []
                        for j in range(batch_size):
                            rgb_vals = QuaternionTensor.batch_to_rgb(results[j])
                            rgb_tensors.append(rgb_vals)
                        
                        # Convert to PIL images
                        for rgb in rgb_tensors:
                            if len(rgb.shape) == 3 and rgb.shape[-1] == 3:
                                img_array = rgb.cpu().numpy()
                                img = Image.fromarray(img_array.astype(np.uint8))
                                images.append(img)
                            else:
                                img = Image.fromarray(np.zeros((self.height, self.width, 3), dtype=np.uint8))
                                images.append(img)
                                
                        # Continue to next batch if vectorized processing succeeded
                        continue
                    except Exception as e:
                        logger.warning(f"Vectorized processing failed, falling back to regular: {e}")
            
            # Regular approach - evaluate each expression
            batch_results = []
            with torch.no_grad():
                for expr in batch:
                    result = expr.evaluate(self.coords)
                    batch_results.append(result.data)
                
                # Convert quaternions to RGB all at once using tensor operations
                rgb_tensors = []
                if len(batch_results) > 0:
                    # Stack results if possible for more efficient conversion
                    try:
                        stacked_results = torch.stack(batch_results)
                        values = torch.clamp(stacked_results[..., 1:4], -10.0, 10.0)
                        rgb_all = 255.0 / (1.0 + torch.exp(-values))
                        rgb_all = rgb_all.to(torch.uint8)
                        
                        # Split back to individual tensors
                        for j in range(batch_size):
                            rgb_tensors.append(rgb_all[j])
                    except Exception:
                        # Fall back to individual processing
                        for result_data in batch_results:
                            rgb = QuaternionTensor.batch_to_rgb(result_data)
                            rgb_tensors.append(rgb)
                
                # Convert to PIL Images efficiently
                for rgb in rgb_tensors:
                    # Ensure RGB tensor has proper shape
                    if len(rgb.shape) == 3 and rgb.shape[-1] == 3:
                        img_array = rgb.cpu().numpy()
                        img = Image.fromarray(img_array.astype(np.uint8))
                        images.append(img)
                    else:
                        # Create blank image as fallback
                        img = Image.fromarray(np.zeros((self.height, self.width, 3), dtype=np.uint8))
                        images.append(img)
                
                # Free up memory on each batch
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
                
        return images


def test_pytorch_main():
    """
    Test function to generate multiple images using the ExpressionNode and ImageGenerator.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    folder_name = f"pytorch_images_{int(time.time())}"
    os.makedirs(folder_name, exist_ok=True)

    image_size = 128
    generator = ImageGenerator(image_size, image_size, device=device)
    num_images = 5
    logger.info(f"Generating {num_images} images...")

    for i in range(num_images):
        try:
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            expr = ExpressionNode.create_random(depth=8)
            logger.info(f"Generated Expression: {expr}")
            img = generator.generate(expr)
            image_path = os.path.join(folder_name, f"image_{i}.png")
            img.save(image_path)
            logger.info(f"Saved {image_path}")
        except Exception as ex:
            logger.error(f"Error generating image {i}: {ex}")

    logger.info(f"Completed generation of {num_images} images.")


if __name__ == "__main__":
    test_pytorch_main()