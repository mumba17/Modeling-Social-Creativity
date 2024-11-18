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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OpType(Enum):
    CONSTANT = auto()
    UNARY = auto()
    BINARY = auto()
    COORDINATE = auto()

@dataclass
class Quaternion:
    w: float = 0.0
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def __add__(self, other: 'Quaternion') -> 'Quaternion':
        return Quaternion(
            self.w + other.w,
            self.x + other.x,
            self.y + other.y,
            self.z + other.z
        )

    def __sub__(self, other: 'Quaternion') -> 'Quaternion':
        return Quaternion(
        self.w - other.w,
        self.x - other.x, 
        self.y - other.y,
        self.z - other.z
        )

    def __mul__(self, other: 'Quaternion') -> 'Quaternion':
        return Quaternion(
            self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z,
            self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y,
            self.w * other.y - self.x * other.z + self.y * other.w + self.z * other.x,
            self.w * other.z + self.x * other.y - self.y * other.x + self.z * other.w
        )
        
        
    def __truediv__(self, other: 'Quaternion') -> 'Quaternion':
        # Divide by conjugate to get reciprocal
        denom = other.norm_squared()
        if denom < 1e-10:
            return Quaternion()
        recip = other.conjugate()
        recip.w /= denom
        recip.x /= denom
        recip.y /= denom 
        recip.z /= denom
        return self * recip
    
    def __neg__(self) -> 'Quaternion':
        """Handle unary minus operator (-q)"""
        return Quaternion(
        -self.w,
        -self.x,
        -self.y,
        -self.z
    )

    def rotate(self, angle: float) -> 'Quaternion':
        """Rotate quaternion by angle (in radians)"""
        c = np.cos(angle/2)
        s = np.sin(angle/2)
        return Quaternion(
            c*self.w - s*self.x,
            c*self.x + s*self.w,
            c*self.y - s*self.z,
            c*self.z + s*self.y
        )
        
    def floor(self) -> 'Quaternion':
        """Floor each component"""
        return Quaternion(
            np.floor(self.w),
            np.floor(self.x),
            np.floor(self.y),
            np.floor(self.z)
        )

    def modulo(self, mod: float) -> 'Quaternion':
        """Modulo each component by mod"""
        return Quaternion(
            self.w % mod,
            self.x % mod,
            self.y % mod,
            self.z % mod
        )

    def conjugate(self) -> 'Quaternion':
        return Quaternion(self.w, -self.x, -self.y, -self.z)

    def norm_squared(self) -> float:
        return self.w**2 + self.x**2 + self.y**2 + self.z**2

    def normalize(self) -> 'Quaternion':
        n = np.sqrt(self.norm_squared())
        if n > 1e-10:
            return Quaternion(
                self.w/n, self.x/n, self.y/n, self.z/n
            )
        return Quaternion()

    def to_rgb(self) -> tuple[int, int, int]:
        def clamp(x: float) -> int:
            try:
                if abs(x) > 30.0:
                    return 0
                return int(255 * (1.0 / (1.0 + np.exp(-x))))
            except Exception:
                return 0
        return (clamp(self.x), clamp(self.y), clamp(self.z))

def q_sin(q: Quaternion) -> Quaternion:
    return Quaternion(
        np.sin(q.w), np.sin(q.x),
        np.sin(q.y), np.sin(q.z)
    )

def q_exp(q: Quaternion) -> Quaternion:
    def safe_exp(x):
        return np.exp(np.clip(x, -30.0, 30.0))
    return Quaternion(
        safe_exp(q.w), safe_exp(q.x),
        safe_exp(q.y), safe_exp(q.z)
    )

def q_log(q: Quaternion) -> Quaternion:
    def safe_log(x):
        return np.log(max(abs(x), 1e-10))
    return Quaternion(
        safe_log(q.w), safe_log(q.x),
        safe_log(q.y), safe_log(q.z)
    )
    
def q_cos(q: Quaternion) -> Quaternion:
    return Quaternion(
        np.cos(q.w), np.cos(q.x),
        np.cos(q.y), np.cos(q.z)
    )

def q_tan(q: Quaternion) -> Quaternion:
    return q_sin(q) * q_cos(q).conjugate()

def q_abs(q: Quaternion) -> Quaternion:
    return Quaternion(
        abs(q.w), abs(q.x),
        abs(q.y), abs(q.z)
    )

def q_sqrt(q: Quaternion) -> Quaternion:
    n = np.sqrt(q.norm_squared())
    if n < 1e-10:
        return Quaternion()
    return q * Quaternion(1/np.sqrt(n))

def q_power(q: Quaternion, p: float) -> Quaternion:
    """Raise quaternion to real power"""
    return q_exp(q_log(q) * Quaternion(p))

def q_inverse(q: Quaternion) -> Quaternion:
    """Get multiplicative inverse of quaternion"""
    n = q.norm_squared() 
    if n < 1e-10:
        return Quaternion()
    c = q.conjugate()
    return Quaternion(c.w/n, c.x/n, c.y/n, c.z/n)

def q_cube(q: Quaternion) -> Quaternion:
    """Cube a quaternion"""
    return q * q * q

def q_sinh(q: Quaternion) -> Quaternion:
    """Hyperbolic sine"""
    return (q_exp(q) - q_exp(-q)) * Quaternion(0.5)

def q_cosh(q: Quaternion) -> Quaternion:
    """Hyperbolic cosine"""  
    return (q_exp(q) + q_exp(-q)) * Quaternion(0.5)

def coord_to_quaternion(x: float, y: float) -> Quaternion:
    return Quaternion(0, x, y, 0)

GOLDEN_RATIO = (1 + np.sqrt(5)) / 2
Q_IDENTITY = Quaternion(1, 0, 0, 0)
Q_I = Quaternion(0, 1, 0, 0)
Q_J = Quaternion(0, 0, 1, 0)
Q_K = Quaternion(0, 0, 0, 1)
Q_ZERO = Quaternion(0, 0, 0, 0)

# Add these new lists right after
CONSTANTS = [
    (lambda: Q_IDENTITY, OpType.CONSTANT),
    (lambda: Q_I, OpType.CONSTANT), 
    (lambda: Q_J, OpType.CONSTANT),
    (lambda: Q_K, OpType.CONSTANT),
    (lambda: Quaternion(GOLDEN_RATIO), OpType.CONSTANT)
]

COORDINATES = [
    (coord_to_quaternion, OpType.COORDINATE),
    (lambda x,y: Quaternion(0, x, 0, 0), OpType.COORDINATE),
    (lambda x,y: Quaternion(0, 0, y, 0), OpType.COORDINATE),
    (lambda x,y: Quaternion(0, y, x, 0), OpType.COORDINATE),
    (lambda x,y: Quaternion(x*y, x, y, 0), OpType.COORDINATE),
    (lambda x,y: Quaternion(x*x - y*y, x, y, 0), OpType.COORDINATE),
    (lambda x,y: Quaternion(np.sin(x)*np.cos(y), x, y, 0), OpType.COORDINATE)
]

# Define available operators
OPERATORS = [
    (operator.add, OpType.BINARY),
    (operator.sub, OpType.BINARY), 
    (operator.mul, OpType.BINARY),
    (operator.truediv, OpType.BINARY),
    (q_inverse, OpType.UNARY), 
    (q_cube, OpType.UNARY),
    (q_sinh, OpType.UNARY),
    (q_cosh, OpType.UNARY),
    (lambda q: q.conjugate(), OpType.UNARY),
    (lambda q: q.normalize(), OpType.UNARY),
    (lambda q: q.rotate(np.pi/4), OpType.UNARY),
    (lambda q: q.floor(), OpType.UNARY),
    (lambda q: q.modulo(2.0), OpType.UNARY),
    (q_sin, OpType.UNARY),
    (q_cos, OpType.UNARY),
    (q_tan, OpType.UNARY),
    (q_exp, OpType.UNARY),
    (q_log, OpType.UNARY),
    (q_abs, OpType.UNARY),
    (q_sqrt, OpType.UNARY),
    (lambda q: q_power(q, 2.0), OpType.UNARY),
    (coord_to_quaternion, OpType.COORDINATE)
]

# Map operators to string representations and back
OPERATOR_TO_STRING = {
    operator.add: '+',
    operator.sub: '-', 
    operator.mul: '*',
    operator.truediv: '/',
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
    lambda q: q.conjugate(): 'conj',
    lambda q: q.normalize(): 'norm',
    lambda q: q.rotate(np.pi/4): 'rot45',
    lambda q: q.floor(): 'floor',
    lambda q: q.modulo(2.0): 'mod2',
    lambda q: q_power(q, 2.0): 'square',
    coord_to_quaternion: 'coord',
    lambda: Quaternion(GOLDEN_RATIO): 'phi'
}

STRING_TO_OPERATOR = {v: k for k, v in OPERATOR_TO_STRING.items()}

class ExpressionNode:
    def __init__(self, 
                 operator: Optional[Tuple[Callable, OpType]] = None,
                 left: Optional['ExpressionNode'] = None,
                 right: Optional['ExpressionNode'] = None):
        self.operator = operator[0] if operator else None
        self.op_type = operator[1] if operator else None
        self.left = left
        self.right = right
        self.value: Optional[Quaternion] = None
        self.constant = False

    def evaluate(self, x: float, y: float) -> Quaternion:
        if self.constant and self.value is not None:
            return self.value

        if self.operator is None:
            return coord_to_quaternion(x, y)

        if self.op_type == OpType.COORDINATE:
            return self.operator(x, y)

        if self.op_type == OpType.CONSTANT:
            if not self.value:
                self.value = self.operator()
                self.constant = True
            return self.value

        left_val = self.left.evaluate(x, y) if self.left else coord_to_quaternion(x, y)

        if self.op_type == OpType.UNARY:
            return self.operator(left_val)

        right_val = self.right.evaluate(x, y) if self.right else coord_to_quaternion(x, y)
        return self.operator(left_val, right_val)

    def mutate(self, rate: float = 0.1) -> None:
        if random.random() < rate:
            new_op = random.choice(OPERATORS)
            self.operator = new_op[0]
            self.op_type = new_op[1]
            self.constant = False
            self.value = None

        if self.left:
            self.left.mutate(rate)
        if self.right:
            self.right.mutate(rate)

    def breed(self, other: 'ExpressionNode') -> 'ExpressionNode':
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
        new_node = ExpressionNode((self.operator, self.op_type) if self.operator else None)
        if self.left:
            new_node.left = self.left._copy()
        if self.right:
            new_node.right = self.right._copy()
        return new_node

    def _random_node(self) -> 'ExpressionNode':
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
        """Create a random expression tree"""
        if depth <= 0:
            return cls((coord_to_quaternion, OpType.COORDINATE))

        # Only select operators that have string mappings
        valid_operators = []
        for op, op_type in OPERATORS:
            if op in OPERATOR_TO_STRING:
                valid_operators.append((op, op_type))

        op = random.choice(valid_operators)
        node = cls(op)

        if op[1] in [OpType.UNARY, OpType.BINARY]:
            node.left = cls.create_random(depth - 1)
            if op[1] == OpType.BINARY:
                node.right = cls.create_random(depth - 1)

        return node
    
    def to_string(self) -> str:
        """Convert expression tree to prefix notation string"""
        if self.operator is None:
            return "coord"
            
        op_str = OPERATOR_TO_STRING.get(self.operator, 'unknown')
        
        if self.op_type == OpType.COORDINATE:
            return f"({op_str})"
            
        if self.op_type == OpType.UNARY:
            left_str = self.left.to_string() if self.left else "coord"
            return f"({op_str} {left_str})"
            
        if self.op_type == OpType.BINARY:
            left_str = self.left.to_string() if self.left else "coord"
            right_str = self.right.to_string() if self.right else "coord"
            return f"({op_str} {left_str} {right_str})"
            
        return "coord"

    @classmethod
    def from_string(cls, formula: str) -> 'ExpressionNode':
        """Parse prefix notation string into expression tree"""
        def parse(tokens: list, index: int) -> tuple['ExpressionNode', int]:
            if index >= len(tokens):
                return cls((coord_to_quaternion, OpType.COORDINATE)), index
                
            token = tokens[index]
            
            if token == '(':
                # Read operator
                op_token = tokens[index + 1]
                if op_token not in STRING_TO_OPERATOR:
                    raise ValueError(f"Unknown operator: {op_token}")
                    
                operator = STRING_TO_OPERATOR[op_token]
                node = cls((operator, OpType.BINARY))  # Default to binary
                
                # Find operator type
                for op, op_type in OPERATORS:
                    if op == operator:
                        node.op_type = op_type
                        break
                        
                if node.op_type == OpType.COORDINATE:
                    return node, index + 3  # Skip past closing paren
                    
                # Parse left child
                node.left, index = parse(tokens, index + 2)
                
                if node.op_type == OpType.BINARY:
                    # Parse right child for binary operators
                    node.right, index = parse(tokens, index)
                    
                # Skip past closing paren
                return node, index + 1
                
            elif token == ')':
                return cls((coord_to_quaternion, OpType.COORDINATE)), index + 1
            else:
                # Handle bare coordinate
                return cls((coord_to_quaternion, OpType.COORDINATE)), index + 1
                
        # Split into tokens, preserving parentheses
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
        """String representation of the expression tree"""
        return self.to_string()

class ImageGenerator:
    def __init__(self, width: int = 256, height: int = 256):
        self.width = width
        self.height = height

    def generate(self, expression: ExpressionNode) -> Image.Image:
        img_array = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        x_center = self.width / 2
        y_center = self.height / 2
        scale = min(self.width, self.height) / 4

        for y in range(self.height):
            qy = (y - y_center) / scale
            for x in range(self.width):
                qx = (x - x_center) / scale
                try:
                    q = expression.evaluate(qx, qy)
                    img_array[y, x] = q.to_rgb()
                except Exception as e:
                    logger.error(f"Error at pixel ({x}, {y}): {e}")
                    img_array[y, x] = (0, 0, 0)

        return Image.fromarray(img_array)

def batch_gen():
    # Create timestamped folder name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"images-{timestamp}"
    
    # Create the folder
    os.makedirs(folder_name, exist_ok=True)
    
    # Number of images to generate
    num_images = 1000  # You can change this to generate more/fewer images
    
    # Dictionary to store formulas
    image_formulas = {}
    
    for i in range(num_images):
        # Create a random expression tree
        expr = ExpressionNode.create_random(depth=8)
        
        # Get the formula string
        formula_str = expr.to_string()
        
        # Store formula in dictionary
        image_formulas[f"image-{i}"] = formula_str
        
        # Generate image
        generator = ImageGenerator(56, 56)
        img = generator.generate(expr)
        
        # Save image
        image_path = os.path.join(folder_name, f"image-{i}.png")
        img.save(image_path)
        print(f"Generated {image_path}")

    # Save formulas to JSON
    json_path = os.path.join(folder_name, "formulas.json")
    with open(json_path, 'w') as f:
        json.dump(image_formulas, f, indent=2)
    print(f"Saved formulas to {json_path}")

    print(f"\nAll files have been saved in the '{folder_name}' folder")
    print("Generated files:")
    print(f"- {num_images} images (image-0.png to image-{num_images-1}.png)")
    print("- formulas.json (contains all formulas in JSON format)")

def main():
    #Generate a single image called "test.png"
    expr = ExpressionNode.create_random(depth=8)
    print("Formula:", expr.to_string())
    
    generator = ImageGenerator(128, 128)
    img = generator.generate(expr)
    img.save("test.png")

if __name__ == "__main__":
    main()