#!/usr/bin/env python3
"""
Pure NumPy AES-CBC-256/128 Neural Network Cryptanalysis System
CPU-based machine learning attack with brute force key generation
No PyTorch dependency - uses pure NumPy for neural networks
"""

import random
import json
import time
import math
import struct
import hashlib
import pickle
import os
import copy
import base64
import string
import itertools
import threading
import queue
from typing import List, Dict, Tuple, Any, Optional, Union
from collections import namedtuple, deque
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

# AES and crypto imports
try:
    from Crypto.Cipher import AES
    from Crypto.Util.Padding import pad, unpad
    from Crypto.Random import get_random_bytes
    CRYPTO_AVAILABLE = True
    print("✅ PyCryptodome available - AES encryption enabled")
except ImportError:
    print("❌ PyCryptodome required! Install with: pip install pycryptodome")
    CRYPTO_AVAILABLE = False

# NumPy for neural networks
try:
    import numpy as np
    NUMPY_AVAILABLE = True
    print("✅ NumPy available - Neural networks enabled")
except ImportError:
    print("❌ NumPy required! Install with: pip install numpy")
    NUMPY_AVAILABLE = False

# Data structures for AES cryptanalysis
@dataclass
class AESConfig:
    key_size: int = 256  # 128 or 256 bits
    block_size: int = 16  # AES block size (always 16 bytes)
    mode: str = "CBC"
    iv_size: int = 16
    
    @property
    def key_bytes(self) -> int:
        return self.key_size // 8
    
    @property 
    def key_hex_length(self) -> int:
        return self.key_bytes * 2

@dataclass
class KeyConstraints:
    """Key generation constraints"""
    min_length: int = 16
    max_length: int = 64
    no_all_same: bool = True
    no_half_patterns: bool = True
    no_repeating: bool = True
    must_be_unique: bool = True
    charset: str = string.ascii_letters + string.digits + "!@#$%^&*()_+-=[]{}|;:,.<>?"

@dataclass
class EncryptedPackage:
    ciphertext: bytes
    iv: bytes
    key_size: int
    algorithm: str = "AES-CBC"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'ciphertext': base64.b64encode(self.ciphertext).decode(),
            'iv': base64.b64encode(self.iv).decode(),
            'key_size': self.key_size,
            'algorithm': self.algorithm
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EncryptedPackage':
        return cls(
            ciphertext=base64.b64decode(data['ciphertext']),
            iv=base64.b64decode(data['iv']),
            key_size=data['key_size'],
            algorithm=data.get('algorithm', 'AES-CBC')
        )

@dataclass
class AttackResult:
    success: bool
    decrypted_data: bytes
    plaintext_preview: str
    key_candidate: str
    accuracy_score: float
    attack_method: str
    time_taken: float
    attempts: int

@dataclass
class MLTrainingData:
    """ML training data for AES key prediction"""
    key_features: List[List[float]] = field(default_factory=list)
    key_targets: List[str] = field(default_factory=list)
    plaintext_features: List[List[float]] = field(default_factory=list)
    success_indicators: List[float] = field(default_factory=list)
    pattern_history: List[Dict[str, Any]] = field(default_factory=list)
    training_losses: List[float] = field(default_factory=list)
    session_count: int = 0
    total_examples: int = 0

# Pure NumPy Neural Network Implementation
class NumpyLayer:
    """Base class for neural network layers"""
    
    def __init__(self):
        self.weights = None
        self.biases = None
        self.last_input = None
        self.last_output = None
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        raise NotImplementedError

class Linear(NumpyLayer):
    """Fully connected linear layer"""
    
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        
        # Xavier initialization
        limit = np.sqrt(6.0 / (input_size + output_size))
        self.weights = np.random.uniform(-limit, limit, (input_size, output_size))
        self.biases = np.zeros((1, output_size))
        
        # For gradients
        self.grad_weights = np.zeros_like(self.weights)
        self.grad_biases = np.zeros_like(self.biases)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        self.last_input = x.copy()
        self.last_output = np.dot(x, self.weights) + self.biases
        return self.last_output
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        # Gradients w.r.t. weights and biases
        self.grad_weights = np.dot(self.last_input.T, grad_output)
        self.grad_biases = np.sum(grad_output, axis=0, keepdims=True)
        
        # Gradient w.r.t. input
        grad_input = np.dot(grad_output, self.weights.T)
        return grad_input

class ReLU(NumpyLayer):
    """ReLU activation function"""
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        self.last_input = x.copy()
        self.last_output = np.maximum(0, x)
        return self.last_output
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        grad_input = grad_output.copy()
        grad_input[self.last_input <= 0] = 0
        return grad_input

class Sigmoid(NumpyLayer):
    """Sigmoid activation function"""
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        # Clip input to prevent overflow
        x_clipped = np.clip(x, -500, 500)
        self.last_output = 1 / (1 + np.exp(-x_clipped))
        return self.last_output
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        sig = self.last_output
        grad_input = grad_output * sig * (1 - sig)
        return grad_input

class Softmax(NumpyLayer):
    """Softmax activation function"""
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        # Numerical stability: subtract max
        x_shifted = x - np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(x_shifted)
        self.last_output = exp_x / np.sum(exp_x, axis=1, keepdims=True)
        return self.last_output
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        # Simplified softmax gradient
        softmax_output = self.last_output
        grad_input = softmax_output * (grad_output - np.sum(grad_output * softmax_output, axis=1, keepdims=True))
        return grad_input

class Dropout(NumpyLayer):
    """Dropout layer for regularization"""
    
    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p
        self.mask = None
        self.training = True
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        if self.training:
            self.mask = np.random.binomial(1, 1 - self.p, x.shape) / (1 - self.p)
            return x * self.mask
        else:
            return x
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        if self.training and self.mask is not None:
            return grad_output * self.mask
        else:
            return grad_output
    
    def eval(self):
        self.training = False
    
    def train(self):
        self.training = True

class NumpyNeuralNetwork:
    """Neural network using pure NumPy"""
    
    def __init__(self, layers: List[NumpyLayer]):
        self.layers = layers
        self.training = True
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def backward(self, grad_output: np.ndarray):
        grad = grad_output
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad
    
    def train(self):
        self.training = True
        for layer in self.layers:
            if hasattr(layer, 'train'):
                layer.train()
    
    def eval(self):
        self.training = False
        for layer in self.layers:
            if hasattr(layer, 'eval'):
                layer.eval()
    
    def get_parameters(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Get all weights and biases"""
        params = []
        for layer in self.layers:
            if hasattr(layer, 'weights') and layer.weights is not None:
                params.append((layer.weights, layer.biases))
        return params
    
    def get_gradients(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Get all gradients"""
        grads = []
        for layer in self.layers:
            if hasattr(layer, 'grad_weights'):
                grads.append((layer.grad_weights, layer.grad_biases))
        return grads

class AdamOptimizer:
    """Adam optimizer implementation"""
    
    def __init__(self, learning_rate: float = 0.001, beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = []  # First moment estimates
        self.v = []  # Second moment estimates
        self.t = 0   # Time step
        self.initialized = False
    
    def update(self, network: NumpyNeuralNetwork):
        """Update network parameters using Adam"""
        self.t += 1
        
        parameters = network.get_parameters()
        gradients = network.get_gradients()
        
        if not self.initialized:
            # Initialize moment estimates
            for weights, biases in parameters:
                self.m.append((np.zeros_like(weights), np.zeros_like(biases)))
                self.v.append((np.zeros_like(weights), np.zeros_like(biases)))
            self.initialized = True
        
        # Update parameters
        for i, ((weights, biases), (grad_w, grad_b)) in enumerate(zip(parameters, gradients)):
            m_w, m_b = self.m[i]
            v_w, v_b = self.v[i]
            
            # Update biased first moment estimate
            m_w = self.beta1 * m_w + (1 - self.beta1) * grad_w
            m_b = self.beta1 * m_b + (1 - self.beta1) * grad_b
            
            # Update biased second raw moment estimate
            v_w = self.beta2 * v_w + (1 - self.beta2) * (grad_w ** 2)
            v_b = self.beta2 * v_b + (1 - self.beta2) * (grad_b ** 2)
            
            # Compute bias-corrected first moment estimate
            m_w_hat = m_w / (1 - self.beta1 ** self.t)
            m_b_hat = m_b / (1 - self.beta1 ** self.t)
            
            # Compute bias-corrected second raw moment estimate
            v_w_hat = v_w / (1 - self.beta2 ** self.t)
            v_b_hat = v_b / (1 - self.beta2 ** self.t)
            
            # Update parameters
            weights -= self.learning_rate * m_w_hat / (np.sqrt(v_w_hat) + self.epsilon)
            biases -= self.learning_rate * m_b_hat / (np.sqrt(v_b_hat) + self.epsilon)
            
            # Update stored moments
            self.m[i] = (m_w, m_b)
            self.v[i] = (v_w, v_b)

def mse_loss(predictions: np.ndarray, targets: np.ndarray) -> Tuple[float, np.ndarray]:
    """Mean squared error loss and gradient"""
    diff = predictions - targets
    loss = np.mean(diff ** 2)
    grad = 2 * diff / predictions.shape[0]
    return loss, grad

def binary_cross_entropy_loss(predictions: np.ndarray, targets: np.ndarray) -> Tuple[float, np.ndarray]:
    """Binary cross entropy loss and gradient"""
    # Clip predictions to prevent log(0)
    predictions = np.clip(predictions, 1e-15, 1 - 1e-15)
    
    loss = -np.mean(targets * np.log(predictions) + (1 - targets) * np.log(1 - predictions))
    grad = (predictions - targets) / (predictions * (1 - predictions)) / predictions.shape[0]
    return loss, grad

def cross_entropy_loss(predictions: np.ndarray, targets: np.ndarray) -> Tuple[float, np.ndarray]:
    """Cross entropy loss and gradient"""
    # Clip predictions to prevent log(0)
    predictions = np.clip(predictions, 1e-15, 1 - 1e-15)
    
    # Convert targets to one-hot if needed
    if targets.ndim == 1:
        num_classes = predictions.shape[1]
        targets_one_hot = np.zeros((targets.shape[0], num_classes))
        targets_one_hot[np.arange(targets.shape[0]), targets.astype(int)] = 1
        targets = targets_one_hot
    
    loss = -np.mean(np.sum(targets * np.log(predictions), axis=1))
    grad = (predictions - targets) / predictions.shape[0]
    return loss, grad

# AES Encryption/Decryption Handler (unchanged)
class AESCryptographyEngine:
    """Handles AES-CBC encryption and decryption operations"""
    
    def __init__(self, config: AESConfig):
        self.config = config
        
    def generate_key(self, constraints: KeyConstraints) -> str:
        """Generate a cryptographically valid key following constraints"""
        while True:
            # Generate random key of appropriate length
            if self.config.key_size == 128:
                key_length = 16  # 16 bytes = 128 bits
            else:  # 256
                key_length = 32  # 32 bytes = 256 bits
            
            # Generate random bytes and convert to hex
            random_bytes = get_random_bytes(key_length)
            key_hex = random_bytes.hex()
            
            # Check constraints
            if self._validate_key_constraints(key_hex, constraints):
                return key_hex
    
    def _validate_key_constraints(self, key: str, constraints: KeyConstraints) -> bool:
        """Validate key against constraints"""
        if constraints.no_all_same:
            if len(set(key)) <= 1:
                return False
        
        if constraints.no_half_patterns:
            mid = len(key) // 2
            if key[:mid] == key[mid:]:
                return False
        
        if constraints.no_repeating:
            # Check for repeating patterns
            for pattern_len in range(2, len(key) // 3):
                pattern = key[:pattern_len]
                if key.startswith(pattern * (len(key) // pattern_len)):
                    return False
        
        return True
    
    def encrypt_data(self, plaintext: str, key_hex: str) -> EncryptedPackage:
        """Encrypt data using AES-CBC with given key"""
        if not CRYPTO_AVAILABLE:
            raise RuntimeError("PyCryptodome not available for encryption")
        
        # Convert hex key to bytes
        key_bytes = bytes.fromhex(key_hex)
        
        # Generate random IV
        iv = get_random_bytes(self.config.iv_size)
        
        # Create cipher
        cipher = AES.new(key_bytes, AES.MODE_CBC, iv)
        
        # Pad plaintext and encrypt
        padded_data = pad(plaintext.encode('utf-8'), AES.block_size)
        ciphertext = cipher.encrypt(padded_data)
        
        return EncryptedPackage(
            ciphertext=ciphertext,
            iv=iv,
            key_size=self.config.key_size,
            algorithm=f"AES-CBC-{self.config.key_size}"
        )
    
    def decrypt_data(self, package: EncryptedPackage, key_hex: str) -> Tuple[bool, str]:
        """Attempt to decrypt data with given key"""
        if not CRYPTO_AVAILABLE:
            return False, "PyCryptodome not available"
        
        try:
            # Convert hex key to bytes
            key_bytes = bytes.fromhex(key_hex)
            
            # Create cipher
            cipher = AES.new(key_bytes, AES.MODE_CBC, package.iv)
            
            # Decrypt
            decrypted_padded = cipher.decrypt(package.ciphertext)
            
            # Remove padding
            decrypted_data = unpad(decrypted_padded, AES.block_size)
            
            # Try to decode as UTF-8
            plaintext = decrypted_data.decode('utf-8')
            
            return True, plaintext
            
        except Exception as e:
            return False, f"Decryption failed: {str(e)}"
    
    def analyze_ciphertext(self, package: EncryptedPackage) -> Dict[str, Any]:
        """Analyze ciphertext for patterns and properties"""
        analysis = {
            'ciphertext_length': len(package.ciphertext),
            'block_count': len(package.ciphertext) // 16,
            'iv_hex': package.iv.hex(),
            'key_size': package.key_size,
            'algorithm': package.algorithm
        }
        
        # Byte frequency analysis
        byte_freq = {}
        for byte in package.ciphertext:
            byte_freq[byte] = byte_freq.get(byte, 0) + 1
        
        analysis['byte_entropy'] = self._calculate_entropy(list(byte_freq.values()))
        analysis['unique_bytes'] = len(byte_freq)
        analysis['most_common_byte'] = max(byte_freq.items(), key=lambda x: x[1])
        
        return analysis
    
    def _calculate_entropy(self, frequencies: List[int]) -> float:
        """Calculate Shannon entropy"""
        total = sum(frequencies)
        if total == 0:
            return 0
        
        entropy = 0
        for freq in frequencies:
            if freq > 0:
                p = freq / total
                entropy -= p * math.log2(p)
        
        return entropy

# Pure NumPy Neural Networks for AES
class AESKeyPredictionNetwork:
    """Neural network for predicting AES key patterns using pure NumPy"""
    
    def __init__(self, input_size=64, hidden_sizes=[512, 256, 128], vocab_size=16):
        self.input_size = input_size
        self.vocab_size = vocab_size
        self.key_length = 32  # Assume 256-bit keys by default
        
        # Create network layers
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(Linear(prev_size, hidden_size))
            layers.append(ReLU())
            layers.append(Dropout(0.3))
            prev_size = hidden_size
        
        # Output layer for each hex character position
        self.feature_network = NumpyNeuralNetwork(layers)
        
        # Separate output heads for each position
        self.output_heads = []
        for _ in range(self.key_length):
            head_layers = [
                Linear(prev_size, vocab_size),
                Softmax()
            ]
            self.output_heads.append(NumpyNeuralNetwork(head_layers))
        
        # Optimizer
        self.optimizer = AdamOptimizer(learning_rate=0.001)
        
    def forward(self, features: np.ndarray) -> List[np.ndarray]:
        """Forward pass"""
        # Process features
        processed = self.feature_network.forward(features)
        
        # Generate predictions for each position
        outputs = []
        for head in self.output_heads:
            output = head.forward(processed)
            outputs.append(output)
        
        return outputs
    
    def predict(self, features: np.ndarray) -> str:
        """Predict a key from features"""
        self.feature_network.eval()
        for head in self.output_heads:
            head.eval()
        
        outputs = self.forward(features)
        
        # Generate key by sampling from each position
        key_chars = []
        for output in outputs:
            # Get probabilities and sample
            probs = output[0]  # Take first (and only) sample
            char_idx = np.random.choice(self.vocab_size, p=probs)
            key_chars.append(format(char_idx, 'x'))
        
        return ''.join(key_chars)
    
    def train_step(self, features: np.ndarray, target_keys: List[str]) -> float:
        """Single training step"""
        self.feature_network.train()
        for head in self.output_heads:
            head.train()
        
        # Convert target keys to one-hot vectors
        targets = []
        for key in target_keys:
            key_targets = []
            for i, char in enumerate(key.lower()[:self.key_length]):
                if char in '0123456789abcdef':
                    target_idx = int(char, 16)
                else:
                    target_idx = 0
                key_targets.append(target_idx)
            
            # Pad if necessary
            while len(key_targets) < self.key_length:
                key_targets.append(0)
            
            targets.append(key_targets)
        
        targets = np.array(targets)
        
        # Forward pass
        outputs = self.forward(features)
        
        # Calculate loss and gradients
        total_loss = 0
        for i, (output, target_pos) in enumerate(zip(outputs, targets.T)):
            target_one_hot = np.zeros((target_pos.shape[0], self.vocab_size))
            target_one_hot[np.arange(target_pos.shape[0]), target_pos] = 1
            
            loss, grad = cross_entropy_loss(output, target_one_hot)
            total_loss += loss
            
            # Backward pass for this head
            self.output_heads[i].backward(grad)
        
        # Update parameters
        for head in self.output_heads:
            self.optimizer.update(head)
        
        return total_loss / len(outputs)

class AESPlaintextClassifier:
    """Neural network to classify if decryption was successful using pure NumPy"""
    
    def __init__(self, input_size=1024):
        # Create network
        layers = [
            Linear(input_size, 512),
            ReLU(),
            Dropout(0.4),
            Linear(512, 256),
            ReLU(),
            Dropout(0.3),
            Linear(256, 128),
            ReLU(),
            Dropout(0.2),
            Linear(128, 64),
            ReLU(),
            Linear(64, 1),
            Sigmoid()
        ]
        
        self.network = NumpyNeuralNetwork(layers)
        self.optimizer = AdamOptimizer(learning_rate=0.001)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        return self.network.forward(x)
    
    def predict(self, features: np.ndarray) -> float:
        """Predict if plaintext is valid"""
        self.network.eval()
        if features.ndim == 1:
            features = features.reshape(1, -1)
        output = self.forward(features)
        return float(output[0, 0])
    
    def train_step(self, features: np.ndarray, targets: np.ndarray) -> float:
        """Single training step"""
        self.network.train()
        
        # Forward pass
        predictions = self.forward(features)
        
        # Calculate loss
        loss, grad = binary_cross_entropy_loss(predictions, targets.reshape(-1, 1))
        
        # Backward pass
        self.network.backward(grad)
        
        # Update parameters
        self.optimizer.update(self.network)
        
        return loss

class AESPatternLearner:
    """Network to learn key generation patterns using pure NumPy"""
    
    def __init__(self, sequence_length=32, vocab_size=16, hidden_size=128):
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        
        # Simple feedforward network for pattern learning
        # (LSTM is complex to implement in pure NumPy, so using feedforward)
        layers = [
            Linear(sequence_length * vocab_size, hidden_size * 2),  # Embed sequence
            ReLU(),
            Dropout(0.3),
            Linear(hidden_size * 2, hidden_size),
            ReLU(),
            Dropout(0.2),
            Linear(hidden_size, vocab_size),
            Softmax()
        ]
        
        self.network = NumpyNeuralNetwork(layers)
        self.optimizer = AdamOptimizer(learning_rate=0.0005)
    
    def _sequence_to_onehot(self, sequence: List[int]) -> np.ndarray:
        """Convert sequence to one-hot encoding"""
        onehot = np.zeros((self.sequence_length, self.vocab_size))
        for i, val in enumerate(sequence[:self.sequence_length]):
            if 0 <= val < self.vocab_size:
                onehot[i, val] = 1
        return onehot.flatten()
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        return self.network.forward(x)
    
    def predict_next_char(self, sequence: List[int]) -> int:
        """Predict next character in sequence"""
        self.network.eval()
        
        # Convert to one-hot and predict
        onehot = self._sequence_to_onehot(sequence).reshape(1, -1)
        output = self.forward(onehot)
        
        # Sample from distribution
        probs = output[0]
        return np.random.choice(self.vocab_size, p=probs)
    
    def train_step(self, sequences: List[List[int]], targets: List[int]) -> float:
        """Single training step"""
        self.network.train()
        
        # Convert sequences to one-hot
        batch_inputs = []
        for seq in sequences:
            onehot = self._sequence_to_onehot(seq)
            batch_inputs.append(onehot)
        
        batch_inputs = np.array(batch_inputs)
        targets_array = np.array(targets)
        
        # Forward pass
        predictions = self.forward(batch_inputs)
        
        # Calculate loss
        loss, grad = cross_entropy_loss(predictions, targets_array)
        
        # Backward pass
        self.network.backward(grad)
        
        # Update parameters
        self.optimizer.update(self.network)
        
        return loss

# Advanced ML Training System for AES (updated for NumPy)
class AESMLTrainingSystem:
    """Machine learning training system for AES key prediction using pure NumPy"""
    
    def __init__(self, config: AESConfig, training_file="aes_numpy_ml_training.pkl"):
        self.config = config
        self.training_file = training_file
        self.training_data = MLTrainingData()
        
        # Initialize networks
        if NUMPY_AVAILABLE:
            self.key_predictor = AESKeyPredictionNetwork(
                input_size=64,
                vocab_size=16
            )
            self.plaintext_classifier = AESPlaintextClassifier()
            self.pattern_learner = AESPatternLearner(
                sequence_length=self.config.key_hex_length
            )
            
            print("🧠 Pure NumPy ML Training System initialized")
        
        # Load existing training data
        self.load_training_data()
    
    def load_training_data(self) -> bool:
        """Load existing training data"""
        try:
            if os.path.exists(self.training_file):
                with open(self.training_file, 'rb') as f:
                    saved_data = pickle.load(f)
                    self.training_data = saved_data.get('training_data', MLTrainingData())
                    
                    print(f"✅ Loaded NumPy ML training data: {self.training_data.total_examples} examples")
                    # Note: Model states not saved for NumPy implementation (would need custom serialization)
                
                return True
            else:
                print("🆕 No existing training data - starting fresh")
                return False
        except Exception as e:
            print(f"⚠️ Error loading training data: {e}")
            return False
    
    def save_training_data(self) -> bool:
        """Save training data"""
        try:
            self.training_data.session_count += 1
            
            save_data = {
                'training_data': self.training_data
                # Note: NumPy model states not saved (would need custom serialization)
            }
            
            with open(self.training_file, 'wb') as f:
                pickle.dump(save_data, f)
            
            print(f"💾 Saved NumPy ML training data: Session #{self.training_data.session_count}")
            return True
        except Exception as e:
            print(f"⚠️ Error saving training data: {e}")
            return False
    
    def extract_key_features(self, key_hex: str) -> List[float]:
        """Extract features from a hex key for ML training"""
        features = []
        
        # Basic statistical features
        char_counts = {}
        for char in '0123456789abcdef':
            char_counts[char] = key_hex.lower().count(char)
        
        features.extend(list(char_counts.values()))  # 16 features
        
        # Pattern features
        features.append(len(set(key_hex.lower())))  # Unique characters
        features.append(max(char_counts.values()))  # Most frequent char count
        features.append(min(char_counts.values()))  # Least frequent char count
        
        # Positional entropy
        positions = [[] for _ in range(16)]
        for i, char in enumerate(key_hex.lower()):
            if char in '0123456789abcdef':
                positions[int(char, 16)].append(i)
        
        position_entropy = 0
        for pos_list in positions:
            if pos_list:
                # Calculate positional spread
                spread = max(pos_list) - min(pos_list) if len(pos_list) > 1 else 0
                position_entropy += spread
        
        features.append(position_entropy / len(key_hex))
        
        # Transition patterns
        transitions = {}
        for i in range(len(key_hex) - 1):
            transition = key_hex[i:i+2].lower()
            transitions[transition] = transitions.get(transition, 0) + 1
        
        features.append(len(transitions))  # Unique transitions
        features.append(max(transitions.values()) if transitions else 0)  # Max transition frequency
        
        # Padding to ensure consistent size
        while len(features) < 64:
            features.append(0.0)
        
        return features[:64]
    
    def extract_plaintext_features(self, data: bytes) -> List[float]:
        """Extract features from plaintext for classification"""
        features = []
        
        try:
            # Try to decode as text
            text = data.decode('utf-8', errors='ignore')
            
            # Text statistics
            features.append(len(text))
            features.append(sum(1 for c in text if c.isalpha()) / max(1, len(text)))
            features.append(sum(1 for c in text if c.isdigit()) / max(1, len(text)))
            features.append(sum(1 for c in text if c.isspace()) / max(1, len(text)))
            features.append(sum(1 for c in text if c.isprintable()) / max(1, len(text)))
            
            # Character frequency entropy
            char_freq = {}
            for char in text:
                char_freq[char] = char_freq.get(char, 0) + 1
            
            if char_freq:
                total_chars = sum(char_freq.values())
                entropy = 0
                for count in char_freq.values():
                    p = count / total_chars
                    entropy -= p * math.log2(p)
                features.append(entropy)
            else:
                features.append(0)
            
        except:
            # If can't decode as text, analyze as bytes
            features.extend([0, 0, 0, 0, 0, 0])
        
        # Byte-level features
        byte_freq = {}
        for byte in data:
            byte_freq[byte] = byte_freq.get(byte, 0) + 1
        
        features.append(len(byte_freq))  # Unique bytes
        features.append(max(byte_freq.values()) if byte_freq else 0)  # Most frequent byte count
        
        # Byte entropy
        if byte_freq:
            total_bytes = sum(byte_freq.values())
            byte_entropy = 0
            for count in byte_freq.values():
                p = count / total_bytes
                byte_entropy -= p * math.log2(p)
            features.append(byte_entropy)
        else:
            features.append(0)
        
        # Padding to ensure consistent size
        while len(features) < 1024:
            features.append(0.0)
        
        return features[:1024]
    
    def add_training_example(self, key_hex: str, success: bool, plaintext: str = ""):
        """Add a training example"""
        key_features = self.extract_key_features(key_hex)
        
        if success and plaintext:
            plaintext_features = self.extract_plaintext_features(plaintext.encode('utf-8'))
        else:
            plaintext_features = [0.0] * 1024
        
        self.training_data.key_features.append(key_features)
        self.training_data.key_targets.append(key_hex)
        self.training_data.plaintext_features.append(plaintext_features)
        self.training_data.success_indicators.append(float(success))
        self.training_data.total_examples += 1
        
        # Keep dataset manageable
        max_examples = 10000
        if len(self.training_data.key_features) > max_examples:
            # Keep most recent examples
            keep_count = max_examples // 2
            self.training_data.key_features = self.training_data.key_features[-keep_count:]
            self.training_data.key_targets = self.training_data.key_targets[-keep_count:]
            self.training_data.plaintext_features = self.training_data.plaintext_features[-keep_count:]
            self.training_data.success_indicators = self.training_data.success_indicators[-keep_count:]
    
    def train_networks(self, epochs=10) -> Dict[str, float]:
        """Train all neural networks"""
        if not NUMPY_AVAILABLE or len(self.training_data.key_features) < 50:
            return {'key_loss': 0, 'classifier_loss': 0, 'pattern_loss': 0}
        
        losses = {'key_loss': 0, 'classifier_loss': 0, 'pattern_loss': 0}
        
        # Train plaintext classifier
        if len(self.training_data.plaintext_features) >= 20:
            classifier_loss = self._train_classifier(epochs)
            losses['classifier_loss'] = classifier_loss
        
        # Train pattern learner
        if len(self.training_data.key_targets) >= 20:
            pattern_loss = self._train_pattern_learner(epochs)
            losses['pattern_loss'] = pattern_loss
        
        # Train key predictor
        if len(self.training_data.key_features) >= 20:
            key_loss = self._train_key_predictor(epochs)
            losses['key_loss'] = key_loss
        
        return losses
    
    def _train_classifier(self, epochs: int) -> float:
        """Train the plaintext classifier"""
        features = np.array(self.training_data.plaintext_features, dtype=np.float32)
        targets = np.array(self.training_data.success_indicators, dtype=np.float32)
        
        total_loss = 0
        batch_size = 16
        
        for epoch in range(epochs):
            # Shuffle data
            indices = np.random.permutation(len(features))
            
            epoch_loss = 0
            num_batches = 0
            
            for i in range(0, len(indices), batch_size):
                batch_indices = indices[i:i+batch_size]
                batch_features = features[batch_indices]
                batch_targets = targets[batch_indices]
                
                loss = self.plaintext_classifier.train_step(batch_features, batch_targets)
                epoch_loss += loss
                num_batches += 1
            
            if num_batches > 0:
                total_loss += epoch_loss / num_batches
        
        return total_loss / epochs if epochs > 0 else 0
    
    def _train_key_predictor(self, epochs: int) -> float:
        """Train the key prediction network"""
        features = np.array(self.training_data.key_features, dtype=np.float32)
        targets = self.training_data.key_targets
        
        total_loss = 0
        batch_size = 16
        
        for epoch in range(epochs):
            # Shuffle data
            indices = np.random.permutation(len(features))
            
            epoch_loss = 0
            num_batches = 0
            
            for i in range(0, len(indices), batch_size):
                batch_indices = indices[i:i+batch_size]
                batch_features = features[batch_indices]
                batch_targets = [targets[j] for j in batch_indices]
                
                loss = self.key_predictor.train_step(batch_features, batch_targets)
                epoch_loss += loss
                num_batches += 1
            
            if num_batches > 0:
                total_loss += epoch_loss / num_batches
        
        return total_loss / epochs if epochs > 0 else 0
    
    def _train_pattern_learner(self, epochs: int) -> float:
        """Train the pattern learning network"""
        # Convert hex keys to sequences of integers
        sequences = []
        targets = []
        
        for key_hex in self.training_data.key_targets:
            sequence = [int(c, 16) for c in key_hex.lower() if c in '0123456789abcdef']
            if len(sequence) >= 2:
                # Create training pairs (sequence -> next character)
                for i in range(len(sequence) - 1):
                    input_seq = sequence[:i+1]
                    target_char = sequence[i+1]
                    
                    # Pad sequence if needed
                    if len(input_seq) < self.config.key_hex_length:
                        input_seq = input_seq + [0] * (self.config.key_hex_length - len(input_seq))
                    
                    sequences.append(input_seq[:self.config.key_hex_length])
                    targets.append(target_char)
        
        if not sequences:
            return 0
        
        total_loss = 0
        batch_size = 16
        
        for epoch in range(epochs):
            # Shuffle data
            indices = np.random.permutation(len(sequences))
            
            epoch_loss = 0
            num_batches = 0
            
            for i in range(0, len(indices), batch_size):
                batch_indices = indices[i:i+batch_size]
                batch_sequences = [sequences[j] for j in batch_indices]
                batch_targets = [targets[j] for j in batch_indices]
                
                loss = self.pattern_learner.train_step(batch_sequences, batch_targets)
                epoch_loss += loss
                num_batches += 1
            
            if num_batches > 0:
                total_loss += epoch_loss / num_batches
        
        return total_loss / epochs if epochs > 0 else 0
    
    def predict_key_candidate(self, ciphertext_features: List[float]) -> str:
        """Predict a key candidate using neural networks"""
        if not NUMPY_AVAILABLE:
            # Fallback: generate random key
            return get_random_bytes(self.config.key_bytes).hex()
        
        try:
            features = np.array(ciphertext_features, dtype=np.float32).reshape(1, -1)
            predicted_key = self.key_predictor.predict(features)
            return predicted_key
        
        except Exception as e:
            print(f"⚠️ Neural prediction failed: {e}")
            # Fallback to random key
            return get_random_bytes(self.config.key_bytes).hex()
    
    def evaluate_plaintext(self, data: bytes) -> float:
        """Evaluate if decrypted data looks like valid plaintext"""
        if not NUMPY_AVAILABLE:
            # Simple heuristic
            try:
                text = data.decode('utf-8')
                printable_ratio = sum(1 for c in text if c.isprintable()) / len(text)
                return printable_ratio
            except:
                return 0.0
        
        try:
            features = np.array(
                self.extract_plaintext_features(data),
                dtype=np.float32
            )
            
            score = self.plaintext_classifier.predict(features)
            return score
        except Exception as e:
            # Fallback to simple heuristic
            try:
                text = data.decode('utf-8', errors='ignore')
                printable_ratio = sum(1 for c in text if c.isprintable()) / max(1, len(text))
                return printable_ratio
            except:
                return 0.0

# Brute Force Key Generator (unchanged)
class BruteForceKeyGenerator:
    """Intelligent brute force key generation"""
    
    def __init__(self, config: AESConfig, constraints: KeyConstraints):
        self.config = config
        self.constraints = constraints
        self.attempted_keys = set()
        
        # Common key patterns to try first
        self.priority_patterns = [
            "common_words", "dates", "patterns", "keyboard_walks", "substitutions"
        ]
        
    def generate_priority_keys(self) -> List[str]:
        """Generate high-priority key candidates"""
        candidates = []
        
        # Common password patterns adapted to hex keys
        common_hex_patterns = [
            "deadbeef", "cafebabe", "feedface", "badc0de", "c0ffee", "facade",
            "decaf", "beef", "babe", "ace", "bad", "cab", "dad", "fad", "fee"
        ]
        
        for pattern in common_hex_patterns:
            # Extend pattern to required length
            extended = self._extend_pattern(pattern, self.config.key_hex_length)
            if self._validate_key(extended):
                candidates.append(extended)
        
        # Date-based patterns
        current_year = 2024
        for year in range(current_year - 10, current_year + 1):
            for month in range(1, 13):
                for day in range(1, 32):
                    date_str = f"{year:04d}{month:02d}{day:02d}"
                    extended = self._extend_pattern(date_str, self.config.key_hex_length)
                    if self._validate_key(extended):
                        candidates.append(extended)
        
        # Sequential patterns
        for start_char in "0123456789abcdef":
            pattern = ""
            for i in range(self.config.key_hex_length):
                char_idx = (int(start_char, 16) + i) % 16
                pattern += format(char_idx, 'x')
            
            if self._validate_key(pattern):
                candidates.append(pattern)
        
        return list(set(candidates))  # Remove duplicates
    
    def _extend_pattern(self, pattern: str, target_length: int) -> str:
        """Extend a pattern to target length"""
        if len(pattern) >= target_length:
            return pattern[:target_length]
        
        # Repeat pattern and add random hex
        extended = pattern
        while len(extended) < target_length:
            if len(extended) + len(pattern) <= target_length:
                extended += pattern
            else:
                # Fill remaining with pattern start or random hex
                remaining = target_length - len(extended)
                extended += pattern[:remaining]
        
        return extended.lower()
    
    def _validate_key(self, key: str) -> bool:
        """Validate key against constraints"""
        if key in self.attempted_keys:
            return False
        
        if not all(c in '0123456789abcdef' for c in key.lower()):
            return False
        
        if len(key) != self.config.key_hex_length:
            return False
        
        # Apply constraints
        if self.constraints.no_all_same and len(set(key)) <= 1:
            return False
        
        if self.constraints.no_half_patterns:
            mid = len(key) // 2
            if key[:mid] == key[mid:]:
                return False
        
        if self.constraints.no_repeating:
            for pattern_len in range(2, len(key) // 3):
                pattern = key[:pattern_len]
                if key == pattern * (len(key) // pattern_len):
                    return False
        
        return True
    
    def generate_random_key(self) -> str:
        """Generate a random key following constraints"""
        max_attempts = 1000
        for _ in range(max_attempts):
            key = get_random_bytes(self.config.key_bytes).hex()
            if self._validate_key(key):
                self.attempted_keys.add(key)
                return key
        
        # Fallback: truly random
        key = get_random_bytes(self.config.key_bytes).hex()
        self.attempted_keys.add(key)
        return key
    
    def generate_mutation(self, base_key: str, mutation_rate: float = 0.1) -> str:
        """Generate a mutation of a successful key"""
        key_chars = list(base_key.lower())
        
        for i in range(len(key_chars)):
            if random.random() < mutation_rate:
                # Mutate this character
                new_char = format(random.randint(0, 15), 'x')
                key_chars[i] = new_char
        
        mutated_key = ''.join(key_chars)
        
        if self._validate_key(mutated_key):
            self.attempted_keys.add(mutated_key)
            return mutated_key
        else:
            return self.generate_random_key()

# Method-Specific Neural Networks
class PriorityPatternNetwork:
    """Neural network specialized for priority pattern prediction"""
    
    def __init__(self, vocab_size=16, key_length=32):
        self.vocab_size = vocab_size
        self.key_length = key_length
        
        # Network for pattern scoring
        layers = [
            Linear(64, 256),  # Pattern features input
            ReLU(),
            Dropout(0.3),
            Linear(256, 128),
            ReLU(),
            Linear(128, 1),
            Sigmoid()
        ]
        
        self.pattern_scorer = NumpyNeuralNetwork(layers)
        self.optimizer = AdamOptimizer(learning_rate=0.001)
        
        # Known successful patterns
        self.successful_patterns = []
        self.pattern_success_rates = {}
    
    def score_pattern(self, pattern_features: np.ndarray) -> float:
        """Score how likely a pattern is to succeed"""
        self.pattern_scorer.eval()
        if pattern_features.ndim == 1:
            pattern_features = pattern_features.reshape(1, -1)
        score = self.pattern_scorer.forward(pattern_features)
        return float(score[0, 0])
    
    def train_on_success(self, pattern_features: np.ndarray, success: bool):
        """Train network on pattern success/failure"""
        self.pattern_scorer.train()
        
        target = np.array([[1.0 if success else 0.0]])
        if pattern_features.ndim == 1:
            pattern_features = pattern_features.reshape(1, -1)
        
        # Forward pass
        prediction = self.pattern_scorer.forward(pattern_features)
        
        # Calculate loss
        loss, grad = binary_cross_entropy_loss(prediction, target)
        
        # Backward pass
        self.pattern_scorer.backward(grad)
        
        # Update parameters
        self.optimizer.update(self.pattern_scorer)
        
        return loss

class BruteForceNetwork:
    """Neural network specialized for brute force key generation"""
    
    def __init__(self, vocab_size=16, key_length=32):
        self.vocab_size = vocab_size
        self.key_length = key_length
        
        # Network for key character prediction
        layers = [
            Linear(80, 512),  # Ciphertext + context features
            ReLU(),
            Dropout(0.4),
            Linear(512, 256),
            ReLU(),
            Dropout(0.3),
            Linear(256, 128),
            ReLU(),
            Linear(128, vocab_size * key_length),  # Predict all positions
            Sigmoid()
        ]
        
        self.key_generator = NumpyNeuralNetwork(layers)
        self.optimizer = AdamOptimizer(learning_rate=0.0005)
        
        # Track successful key patterns
        self.successful_keys = []
        self.char_success_rates = np.zeros((key_length, vocab_size))
        self.char_attempt_counts = np.zeros((key_length, vocab_size))
    
    def generate_key_probabilities(self, context_features: np.ndarray) -> np.ndarray:
        """Generate probability distribution for key characters"""
        self.key_generator.eval()
        if context_features.ndim == 1:
            context_features = context_features.reshape(1, -1)
        
        output = self.key_generator.forward(context_features)
        # Reshape to (key_length, vocab_size)
        probs = output.reshape(self.key_length, self.vocab_size)
        
        # Normalize each position to be a valid probability distribution
        for i in range(self.key_length):
            probs[i] = probs[i] / np.sum(probs[i])
        
        return probs
    
    def sample_key(self, context_features: np.ndarray) -> str:
        """Sample a key from learned distribution"""
        probs = self.generate_key_probabilities(context_features)
        
        key_chars = []
        for i in range(self.key_length):
            char_idx = np.random.choice(self.vocab_size, p=probs[i])
            key_chars.append(format(char_idx, 'x'))
        
        return ''.join(key_chars)
    
    def train_on_key(self, context_features: np.ndarray, key: str, success: bool):
        """Train network on key success/failure"""
        self.key_generator.train()
        
        # Convert key to target distribution
        target = np.zeros((self.key_length, self.vocab_size))
        for i, char in enumerate(key.lower()[:self.key_length]):
            if char in '0123456789abcdef':
                char_idx = int(char, 16)
                target[i, char_idx] = 1.0 if success else 0.1  # Reduce probability if failed
        
        target = target.flatten()
        
        if context_features.ndim == 1:
            context_features = context_features.reshape(1, -1)
        
        # Forward pass
        prediction = self.key_generator.forward(context_features)
        
        # Calculate loss
        loss, grad = mse_loss(prediction, target.reshape(1, -1))
        
        # Backward pass
        self.key_generator.backward(grad)
        
        # Update parameters
        self.optimizer.update(self.key_generator)
        
        # Update character statistics
        for i, char in enumerate(key.lower()[:self.key_length]):
            if char in '0123456789abcdef':
                char_idx = int(char, 16)
                self.char_attempt_counts[i, char_idx] += 1
                if success:
                    self.char_success_rates[i, char_idx] += 1
        
        return loss

class MLGuidedNetwork:
    """Neural network for ML-guided key prediction with cross-method learning"""
    
    def __init__(self, vocab_size=16, key_length=32):
        self.vocab_size = vocab_size
        self.key_length = key_length
        
        # Enhanced network that learns from other methods
        layers = [
            Linear(128, 512),  # Larger input for cross-method features
            ReLU(),
            Dropout(0.3),
            Linear(512, 256),
            ReLU(),
            Dropout(0.2),
            Linear(256, 128),
            ReLU(),
            Linear(128, vocab_size * key_length),
            Softmax()
        ]
        
        self.cross_method_learner = NumpyNeuralNetwork(layers)
        self.optimizer = AdamOptimizer(learning_rate=0.001)
        
        # Cross-method learning data
        self.method_success_data = {
            'priority': [],
            'brute_force': [],
            'pattern': [],
            'mutation': []
        }
    
    def predict_key_from_methods(self, features: np.ndarray, method_successes: Dict[str, float]) -> str:
        """Predict key incorporating knowledge from all methods"""
        self.cross_method_learner.eval()
        
        # Combine features with method success rates
        method_features = np.array([
            method_successes.get('priority', 0.0),
            method_successes.get('brute_force', 0.0),
            method_successes.get('pattern', 0.0),
            method_successes.get('mutation', 0.0)
        ])
        
        # Pad features if needed
        while len(features) < 124:
            features = np.append(features, 0.0)
        features = features[:124]
        
        combined_features = np.concatenate([features, method_features]).reshape(1, -1)
        
        # Generate key
        output = self.cross_method_learner.forward(combined_features)
        probs = output.reshape(self.key_length, self.vocab_size)
        
        key_chars = []
        for i in range(self.key_length):
            char_idx = np.random.choice(self.vocab_size, p=probs[i])
            key_chars.append(format(char_idx, 'x'))
        
        return ''.join(key_chars)
    
    def learn_from_method_success(self, features: np.ndarray, key: str, method: str, success: bool):
        """Learn from success of other methods"""
        # Store cross-method learning data
        self.method_success_data[method].append({
            'features': features.copy(),
            'key': key,
            'success': success
        })
        
        # Train on this example
        if success:
            self._train_on_successful_key(features, key, method)
    
    def _train_on_successful_key(self, features: np.ndarray, key: str, method: str):
        """Train network on successful key from any method"""
        self.cross_method_learner.train()
        
        # Create target distribution favoring successful key
        target = np.zeros((self.key_length, self.vocab_size))
        for i, char in enumerate(key.lower()[:self.key_length]):
            if char in '0123456789abcdef':
                char_idx = int(char, 16)
                target[i, char_idx] = 1.0
        
        # Method success rates as additional features
        method_success_rates = {}
        for m, data in self.method_success_data.items():
            if data:
                success_rate = sum(1 for d in data if d['success']) / len(data)
                method_success_rates[m] = success_rate
            else:
                method_success_rates[m] = 0.0
        
        # Prepare input
        method_features = np.array([
            method_success_rates.get('priority', 0.0),
            method_success_rates.get('brute_force', 0.0),
            method_success_rates.get('pattern', 0.0),
            method_success_rates.get('mutation', 0.0)
        ])
        
        while len(features) < 124:
            features = np.append(features, 0.0)
        features = features[:124]
        
        combined_features = np.concatenate([features, method_features]).reshape(1, -1)
        
        # Forward pass
        prediction = self.cross_method_learner.forward(combined_features)
        
        # Calculate loss
        loss, grad = cross_entropy_loss(prediction, target.flatten().reshape(1, -1))
        
        # Backward pass
        self.cross_method_learner.backward(grad)
        
        # Update parameters
        self.optimizer.update(self.cross_method_learner)
        
        return loss

class EnsembleMetaLearner:
    """Meta-learning system that coordinates all methods and learns from their success rates"""
    
    def __init__(self):
        # Method performance tracking
        self.method_stats = {
            'priority_pattern': {'attempts': 0, 'successes': 0, 'partial_successes': 0},
            'brute_force': {'attempts': 0, 'successes': 0, 'partial_successes': 0},
            'ml_guided': {'attempts': 0, 'successes': 0, 'partial_successes': 0},
            'pattern_mutation': {'attempts': 0, 'successes': 0, 'partial_successes': 0},
            'cross_method': {'attempts': 0, 'successes': 0, 'partial_successes': 0}
        }
        
        # Dynamic method weights based on performance
        self.method_weights = {
            'priority_pattern': 0.25,
            'brute_force': 0.20,
            'ml_guided': 0.25,
            'pattern_mutation': 0.15,
            'cross_method': 0.15
        }
        
        # Success rate predictor network
        layers = [
            Linear(32, 128),  # Method statistics + context
            ReLU(),
            Dropout(0.2),
            Linear(128, 64),
            ReLU(),
            Linear(64, 5),  # Predict success probability for each method
            Softmax()
        ]
        
        self.success_predictor = NumpyNeuralNetwork(layers)
        self.optimizer = AdamOptimizer(learning_rate=0.002)
        
        # Knowledge sharing between methods
        self.shared_knowledge = {
            'successful_patterns': [],
            'successful_mutations': [],
            'effective_features': [],
            'cross_correlations': {}
        }
    
    def update_method_performance(self, method: str, success: bool, accuracy: float):
        """Update performance statistics for a method"""
        if method in self.method_stats:
            self.method_stats[method]['attempts'] += 1
            
            if success and accuracy > 0.9:
                self.method_stats[method]['successes'] += 1
            elif accuracy > 0.3:
                self.method_stats[method]['partial_successes'] += 1
    
    def get_method_success_rates(self) -> Dict[str, float]:
        """Calculate current success rates for each method"""
        success_rates = {}
        
        for method, stats in self.method_stats.items():
            if stats['attempts'] > 0:
                total_success = stats['successes'] + (0.5 * stats['partial_successes'])
                success_rates[method] = total_success / stats['attempts']
            else:
                success_rates[method] = 0.0
        
        return success_rates
    
    def predict_best_method(self, context_features: np.ndarray) -> str:
        """Predict which method is most likely to succeed"""
        self.success_predictor.eval()
        
        # Prepare input features
        success_rates = self.get_method_success_rates()
        method_features = []
        
        for method in ['priority_pattern', 'brute_force', 'ml_guided', 'pattern_mutation', 'cross_method']:
            stats = self.method_stats[method]
            method_features.extend([
                success_rates[method],
                stats['attempts'],
                stats['successes'],
                stats['partial_successes'],
                self.method_weights[method]
            ])
        
        # Pad context features
        while len(context_features) < 7:
            context_features = np.append(context_features, 0.0)
        context_features = context_features[:7]
        
        combined_input = np.concatenate([method_features, context_features]).reshape(1, -1)
        
        # Predict method success probabilities
        method_probs = self.success_predictor.forward(combined_input)[0]
        
        # Return method with highest predicted success
        method_names = ['priority_pattern', 'brute_force', 'ml_guided', 'pattern_mutation', 'cross_method']
        best_method_idx = np.argmax(method_probs)
        
        return method_names[best_method_idx]
    
    def update_method_weights(self):
        """Update method weights based on recent performance"""
        success_rates = self.get_method_success_rates()
        
        # Softmax normalization with temperature
        temperature = 0.5
        total_exp = 0
        
        for method in self.method_weights:
            if method in success_rates:
                self.method_weights[method] = math.exp(success_rates[method] / temperature)
                total_exp += self.method_weights[method]
        
        # Normalize
        if total_exp > 0:
            for method in self.method_weights:
                self.method_weights[method] /= total_exp
        
        # Ensure minimum exploration
        min_weight = 0.05
        for method in self.method_weights:
            if self.method_weights[method] < min_weight:
                self.method_weights[method] = min_weight
        
        # Renormalize
        total_weight = sum(self.method_weights.values())
        for method in self.method_weights:
            self.method_weights[method] /= total_weight
    
    def share_successful_knowledge(self, method: str, key: str, features: List[float], accuracy: float):
        """Share successful attempts across all methods"""
        if accuracy > 0.5:
            knowledge_entry = {
                'method': method,
                'key': key,
                'features': features.copy(),
                'accuracy': accuracy,
                'timestamp': time.time()
            }
            
            if method == 'priority_pattern':
                self.shared_knowledge['successful_patterns'].append(knowledge_entry)
            elif method in ['pattern_mutation', 'ml_guided']:
                self.shared_knowledge['successful_mutations'].append(knowledge_entry)
            
            self.shared_knowledge['effective_features'].append(knowledge_entry)
            
            # Keep only recent successful attempts
            max_knowledge = 1000
            for key in self.shared_knowledge:
                if len(self.shared_knowledge[key]) > max_knowledge:
                    # Keep most recent and highest accuracy
                    self.shared_knowledge[key].sort(key=lambda x: (x['accuracy'], x['timestamp']), reverse=True)
                    self.shared_knowledge[key] = self.shared_knowledge[key][:max_knowledge//2]

# Enhanced Main AES Cryptanalysis Attack System
class AdvancedEnsembleAESAttacker:
    """Advanced ensemble system using all methods with individual neural networks"""
    
    def __init__(self, target_package: EncryptedPackage, known_plaintext: str = ""):
        self.target_package = target_package
        self.known_plaintext = known_plaintext
        
        # Initialize base components
        self.config = AESConfig(key_size=target_package.key_size)
        self.constraints = KeyConstraints()
        self.crypto_engine = AESCryptographyEngine(self.config)
        self.brute_force = BruteForceKeyGenerator(self.config, self.constraints)
        
        # Initialize method-specific neural networks
        self.priority_network = PriorityPatternNetwork(key_length=self.config.key_hex_length)
        self.brute_force_network = BruteForceNetwork(key_length=self.config.key_hex_length)
        self.ml_guided_network = MLGuidedNetwork(key_length=self.config.key_hex_length)
        
        # Meta-learning system
        self.meta_learner = EnsembleMetaLearner()
        
        # Original ML system for feature extraction
        self.ml_system = AESMLTrainingSystem(self.config)
        
        # Attack statistics
        self.attempts = 0
        self.start_time = time.time()
        self.best_result = None
        self.attack_history = []
        self.method_attempts = {method: 0 for method in self.meta_learner.method_weights.keys()}
        
        print(f"🎯 Advanced Ensemble AES Cryptanalysis Attacker initialized")
        print(f"   Target: AES-CBC-{target_package.key_size}")
        print(f"   🧠 Method-specific neural networks: 3")
        print(f"   🎯 Meta-learning coordination system")
        print(f"   🔄 Cross-method knowledge sharing")
        print(f"   📊 Adaptive method weighting")
        
        # Analyze target
        self.ciphertext_analysis = self.crypto_engine.analyze_ciphertext(target_package)
        print(f"   Byte entropy: {self.ciphertext_analysis['byte_entropy']:.3f}")
        print(f"   Unique bytes: {self.ciphertext_analysis['unique_bytes']}")
        
        # Extract context features for meta-learning
        self.context_features = self._extract_context_features()
    
    
    def _extract_context_features(self) -> np.ndarray:
        """Extract context features for meta-learning"""
        features = []
        
        # Ciphertext analysis features
        features.extend([
            self.ciphertext_analysis['byte_entropy'],
            self.ciphertext_analysis['unique_bytes'] / 256.0,
            len(self.target_package.ciphertext) / 1000.0,
            self.config.key_size / 256.0
        ])
        
        # Known plaintext features
        if self.known_plaintext:
            features.extend([
                len(self.known_plaintext) / 100.0,
                sum(1 for c in self.known_plaintext if c.isalpha()) / max(1, len(self.known_plaintext)),
                1.0  # Has known plaintext
            ])
        else:
            features.extend([0.0, 0.0, 0.0])
        
        return np.array(features)
    
    def generate_priority_key(self) -> Tuple[str, str]:
        """Generate key using priority patterns with neural network guidance"""
        method = "priority_pattern"
        self.method_attempts[method] += 1
        
        # Get priority patterns
        priority_keys = self.brute_force.generate_priority_keys()
        
        if not priority_keys:
            return self.brute_force.generate_random_key(), method
        
        # Score patterns using neural network
        scored_patterns = []
        for key in priority_keys[:100]:  # Limit for performance
            pattern_features = self.ml_system.extract_key_features(key)
            
            # Pad features to 64
            while len(pattern_features) < 64:
                pattern_features.append(0.0)
            pattern_features = pattern_features[:64]
            
            score = self.priority_network.score_pattern(np.array(pattern_features))
            scored_patterns.append((key, score))
        
        # Sort by score and select top candidate
        scored_patterns.sort(key=lambda x: x[1], reverse=True)
        
        if scored_patterns:
            return scored_patterns[0][0], method
        else:
            return self.brute_force.generate_random_key(), method
    
    def generate_brute_force_key(self) -> Tuple[str, str]:
        """Generate key using brute force neural network"""
        method = "brute_force"
        self.method_attempts[method] += 1
        
        # Prepare context features for neural network
        context_features = []
        context_features.extend(self.context_features.tolist())
        
        # Add recent performance data
        success_rates = self.meta_learner.get_method_success_rates()
        context_features.extend([
            success_rates.get('priority_pattern', 0.0),
            success_rates.get('brute_force', 0.0),
            success_rates.get('ml_guided', 0.0)
        ])
        
        # Add ciphertext features
        ciphertext_features = self.ml_system.extract_key_features(
            self.target_package.ciphertext.hex()[:self.config.key_hex_length]
        )
        context_features.extend(ciphertext_features[:60])  # Add first 60 features
        
        # Pad to required size
        while len(context_features) < 80:
            context_features.append(0.0)
        context_features = context_features[:80]
        
        # Generate key using neural network
        key = self.brute_force_network.sample_key(np.array(context_features))
        
        return key, method
    
    def generate_ml_guided_key(self) -> Tuple[str, str]:
        """Generate key using ML-guided network with cross-method learning"""
        method = "ml_guided"
        self.method_attempts[method] += 1
        
        # Prepare enhanced features including cross-method knowledge
        features = []
        features.extend(self.context_features.tolist())
        
        # Add method success rates
        success_rates = self.meta_learner.get_method_success_rates()
        features.extend(list(success_rates.values()))
        
        # Add features from successful attempts
        if self.meta_learner.shared_knowledge['effective_features']:
            recent_effective = self.meta_learner.shared_knowledge['effective_features'][-5:]
            avg_features = np.mean([ef['features'][:10] for ef in recent_effective], axis=0)
            features.extend(avg_features.tolist())
        else:
            features.extend([0.0] * 10)
        
        # Add ciphertext analysis
        ciphertext_features = self.ml_system.extract_key_features(
            self.target_package.ciphertext.hex()[:self.config.key_hex_length]
        )
        features.extend(ciphertext_features[:100])
        
        # Generate key using cross-method learning
        key = self.ml_guided_network.predict_key_from_methods(
            np.array(features), success_rates
        )
        
        return key, method
    
    def generate_pattern_mutation_key(self) -> Tuple[str, str]:
        """Generate key by mutating successful patterns"""
        method = "pattern_mutation"
        self.method_attempts[method] += 1
        
        # Get successful keys from shared knowledge
        successful_keys = []
        
        # Collect from all sources
        if self.meta_learner.shared_knowledge['successful_patterns']:
            successful_keys.extend([sp['key'] for sp in self.meta_learner.shared_knowledge['successful_patterns']])
        
        if self.meta_learner.shared_knowledge['successful_mutations']:
            successful_keys.extend([sm['key'] for sm in self.meta_learner.shared_knowledge['successful_mutations']])
        
        if self.best_result and self.best_result.accuracy_score > 0.3:
            successful_keys.append(self.best_result.key_candidate)
        
        if successful_keys:
            # Select base key weighted by success
            base_key = random.choice(successful_keys)
            
            # Generate mutation
            mutated_key = self.brute_force.generate_mutation(base_key, mutation_rate=0.15)
            return mutated_key, method
        else:
            # Fallback to random if no successful keys yet
            return self.brute_force.generate_random_key(), method
    
    def generate_cross_method_key(self) -> Tuple[str, str]:
        """Generate key using insights from all methods"""
        method = "cross_method"
        self.method_attempts[method] += 1
        
        # Combine insights from all neural networks
        features = []
        features.extend(self.context_features.tolist())
        
        # Get predictions from all method networks
        context_features_80 = np.zeros(80)
        context_features_80[:len(self.context_features)] = self.context_features
        
        # Priority network insight
        if len(self.priority_network.successful_patterns) > 0:
            avg_pattern_features = np.mean([
                self.ml_system.extract_key_features(pattern)[:64] 
                for pattern in self.priority_network.successful_patterns[-10:]
            ], axis=0)
            features.extend(avg_pattern_features[:20].tolist())
        else:
            features.extend([0.0] * 20)
        
        # Brute force network character preferences
        if np.sum(self.brute_force_network.char_success_rates) > 0:
            char_prefs = np.mean(self.brute_force_network.char_success_rates, axis=0)
            features.extend(char_prefs.tolist())  # 16 features
        else:
            features.extend([0.0] * 16)
        
        # Method success rates
        success_rates = self.meta_learner.get_method_success_rates()
        features.extend(list(success_rates.values()))
        
        # Generate key using ML-guided network with all insights
        key = self.ml_guided_network.predict_key_from_methods(
            np.array(features), success_rates
        )
        
        return key, method
    
    def attempt_decryption_with_learning(self, key_hex: str, method: str) -> AttackResult:
        """Attempt decryption and train all relevant neural networks"""
        self.attempts += 1
        start_time = time.time()
        
        # Try decryption
        success, result = self.crypto_engine.decrypt_data(self.target_package, key_hex)
        
        if success:
            # Evaluate result quality
            if self.known_plaintext:
                accuracy = self._calculate_plaintext_accuracy(result, self.known_plaintext)
            else:
                accuracy = self.ml_system.evaluate_plaintext(result.encode('utf-8'))
            
            preview = result[:100] + "..." if len(result) > 100 else result
        else:
            accuracy = 0.0
            preview = f"Decryption failed: {result}"
        
        attack_result = AttackResult(
            success=success,
            decrypted_data=result.encode('utf-8') if success else b"",
            plaintext_preview=preview,
            key_candidate=key_hex,
            accuracy_score=accuracy,
            attack_method=method,
            time_taken=time.time() - start_time,
            attempts=self.attempts
        )
        
        # Update meta-learner performance tracking
        self.meta_learner.update_method_performance(method, success and accuracy > 0.9, accuracy)
        
        # Extract features for neural network training
        key_features = self.ml_system.extract_key_features(key_hex)
        while len(key_features) < 64:
            key_features.append(0.0)
        key_features = key_features[:64]
        
        context_features_80 = np.zeros(80)
        context_features_80[:len(self.context_features)] = self.context_features
        
        # Train method-specific neural networks
        if method == "priority_pattern":
            loss = self.priority_network.train_on_success(np.array(key_features), success and accuracy > 0.5)
            if success and accuracy > 0.5:
                self.priority_network.successful_patterns.append(key_hex)
        
        elif method == "brute_force":
            loss = self.brute_force_network.train_on_key(context_features_80, key_hex, success and accuracy > 0.5)
        
        elif method in ["ml_guided", "cross_method"]:
            success_rates = self.meta_learner.get_method_success_rates()
            self.ml_guided_network.learn_from_method_success(
                np.array(key_features), key_hex, method, success and accuracy > 0.5
            )
        
        elif method == "pattern_mutation":
            # Train the ML-guided network since mutations inform ML
            success_rates = self.meta_learner.get_method_success_rates()
            self.ml_guided_network.learn_from_method_success(
                np.array(key_features), key_hex, "mutation", success and accuracy > 0.5
            )
        
        # Share knowledge if successful
        if accuracy > 0.3:
            self.meta_learner.share_successful_knowledge(method, key_hex, key_features, accuracy)
        
        # Update original ML system for compatibility
        self.ml_system.add_training_example(
            key_hex=key_hex,
            success=success and accuracy > 0.8,
            plaintext=result if success else ""
        )
        
        # Track best result
        if self.best_result is None or accuracy > self.best_result.accuracy_score:
            self.best_result = attack_result
        
        self.attack_history.append(attack_result)
        
        return attack_result
    
    def _calculate_plaintext_accuracy(self, decrypted: str, known: str) -> float:
        """Calculate accuracy of decrypted text against known plaintext"""
        if not decrypted or not known:
            return 0.0
        
        min_len = min(len(decrypted), len(known))
        if min_len == 0:
            return 0.0
        
        matches = sum(1 for i in range(min_len) if decrypted[i] == known[i])
        char_accuracy = matches / min_len
        
        length_penalty = abs(len(decrypted) - len(known)) / max(len(decrypted), len(known))
        accuracy = char_accuracy * (1 - length_penalty)
        return max(0.0, min(1.0, accuracy))
    
    def adaptive_ensemble_attack(self, max_attempts: int = 20000, verbose: bool = True) -> AttackResult:
        """Advanced ensemble attack using all methods with continuous learning"""
        print(f"\n🧠 Starting ADAPTIVE ENSEMBLE ATTACK")
        print(f"🎯 All methods with individual neural networks")
        print(f"🔄 Continuous cross-method learning")
        print(f"📊 Meta-learning coordination")
        print(f"⚡ Max attempts: {max_attempts:,}")
        
        phase = 1
        phase_size = 1000
        training_interval = 100
        last_training = 0
        
        # Method selection counters
        method_selections = {method: 0 for method in self.meta_learner.method_weights.keys()}
        
        for i in range(max_attempts):
            # Progress reporting
            if verbose and i % 1000 == 0 and i > 0:
                elapsed = time.time() - self.start_time
                rate = self.attempts / max(1, elapsed)
                
                print(f"\n📊 Progress Report - Attempt {i:,}")
                print(f"   ⚡ Rate: {rate:.1f} attempts/sec")
                print(f"   🏆 Best accuracy: {self.best_result.accuracy_score:.1%}" if self.best_result else "   No success yet")
                
                # Method performance
                success_rates = self.meta_learner.get_method_success_rates()
                print(f"   📈 Method success rates:")
                for method, rate in success_rates.items():
                    attempts = self.method_attempts[method]
                    print(f"     • {method}: {rate:.1%} ({attempts} attempts)")
                
                # Current method weights
                print(f"   ⚖️ Method weights:")
                for method, weight in self.meta_learner.method_weights.items():
                    print(f"     • {method}: {weight:.1%}")
            
            # Determine best method using meta-learner
            if i > 50:  # After some initial data
                predicted_best_method = self.meta_learner.predict_best_method(self.context_features)
            else:
                predicted_best_method = random.choice(list(self.meta_learner.method_weights.keys()))
            
            # Method selection with some randomness
            if random.random() < 0.7:  # 70% follow prediction
                selected_method = predicted_best_method
            else:  # 30% explore
                methods = list(self.meta_learner.method_weights.keys())
                weights = list(self.meta_learner.method_weights.values())
                selected_method = random.choices(methods, weights=weights, k=1)[0]
            
            method_selections[selected_method] += 1
            
            # Generate key based on selected method
            if selected_method == "priority_pattern":
                key, method = self.generate_priority_key()
            elif selected_method == "brute_force":
                key, method = self.generate_brute_force_key()
            elif selected_method == "ml_guided":
                key, method = self.generate_ml_guided_key()
            elif selected_method == "pattern_mutation":
                key, method = self.generate_pattern_mutation_key()
            else:  # cross_method
                key, method = self.generate_cross_method_key()
            
            # Attempt decryption with learning
            result = self.attempt_decryption_with_learning(key, method)
            
            # Check for success
            if result.success and result.accuracy_score > 0.95:
                print(f"\n🎉 ENSEMBLE ATTACK SUCCESS!")
                print(f"🤖 Winning method: {result.attack_method}")
                print(f"✅ Perfect decryption: '{result.decrypted_message.decode()}'")
                print(f"🧠 Method used neural networks and cross-learning")
                return result
            
            # Periodic training and adaptation
            if i - last_training >= training_interval:
                # Update method weights based on performance
                self.meta_learner.update_method_weights()
                
                # Train original ML system occasionally
                if len(self.ml_system.training_data.key_features) > 50:
                    losses = self.ml_system.train_networks(epochs=3)
                
                last_training = i
                
                if verbose and i > 1000:
                    print(f"   🧠 Neural networks updated - continuing with enhanced intelligence")
        
        print(f"\n📊 Final method selection statistics:")
        for method, count in method_selections.items():
            percentage = (count / sum(method_selections.values())) * 100
            print(f"   • {method}: {count} times ({percentage:.1f}%)")
        
        return self.best_result
    
    def unlimited_ensemble_attack(self, verbose: bool = True) -> AttackResult:
        """Unlimited ensemble attack that runs until success"""
        print(f"\n🚀 Starting UNLIMITED ENSEMBLE ATTACK")
        print(f"🧠 All methods with neural networks")
        print(f"🔄 Runs until success or interruption")
        print("⚠️  Press Ctrl+C to interrupt")
        
        cycle = 1
        cycle_size = 5000
        
        try:
            while True:
                if verbose:
                    print(f"\n🔄 Ensemble Cycle {cycle}")
                    print(f"   🎯 Target: {cycle_size:,} attempts this cycle")
                    
                    if self.best_result:
                        print(f"   🏆 Current best: {self.best_result.accuracy_score:.1%}")
                    
                    # Show method performance
                    success_rates = self.meta_learner.get_method_success_rates()
                    best_method = max(success_rates.items(), key=lambda x: x[1])
                    print(f"   🥇 Best method: {best_method[0]} ({best_method[1]:.1%} success)")
                
                # Run adaptive attack for this cycle
                result = self.adaptive_ensemble_attack(max_attempts=cycle_size, verbose=False)
                
                if result and result.success and result.accuracy_score > 0.95:
                    print(f"\n🎉 UNLIMITED ENSEMBLE SUCCESS!")
                    print(f"   Cycle: {cycle}")
                    print(f"   Total attempts: {self.attempts:,}")
                    return result
                
                cycle += 1
                
                # Increase cycle size over time
                if cycle > 5:
                    cycle_size = min(10000, int(cycle_size * 1.1))
        
        except KeyboardInterrupt:
            print(f"\n⚠️ Unlimited ensemble attack interrupted")
            print(f"   Completed {cycle} cycles, {self.attempts:,} total attempts")
            return self.best_result
    
    def get_ensemble_summary(self) -> Dict[str, Any]:
        """Get comprehensive ensemble attack summary"""
        elapsed_time = time.time() - self.start_time
        
        return {
            'total_attempts': self.attempts,
            'elapsed_time': elapsed_time,
            'attempts_per_second': self.attempts / max(1, elapsed_time),
            'best_result': self.best_result,
            'success': self.best_result.success if self.best_result else False,
            'best_accuracy': self.best_result.accuracy_score if self.best_result else 0.0,
            'method_attempts': self.method_attempts.copy(),
            'method_success_rates': self.meta_learner.get_method_success_rates(),
            'method_weights': self.meta_learner.method_weights.copy(),
            'shared_knowledge_size': {
                'patterns': len(self.meta_learner.shared_knowledge['successful_patterns']),
                'mutations': len(self.meta_learner.shared_knowledge['successful_mutations']),
                'features': len(self.meta_learner.shared_knowledge['effective_features'])
            },
            'neural_networks_trained': 3,
            'target_analysis': self.ciphertext_analysis
        }
    
    def save_progress(self):
        """Save all ensemble progress"""
        return self.ml_system.save_training_data()
    
    def priority_attack(self, max_attempts: int = 1000) -> AttackResult:
        """Attack using priority key patterns"""
        print(f"\n🎯 Starting priority pattern attack...")
        
        priority_keys = self.brute_force.generate_priority_keys()
        print(f"   Generated {len(priority_keys)} priority key candidates")
        
        for i, key_candidate in enumerate(priority_keys[:max_attempts]):
            if i % 100 == 0 and i > 0:
                print(f"   Tried {i} priority keys...")
            
            result = self.attempt_decryption(key_candidate, "priority_pattern")
            
            if result.success and result.accuracy_score > 0.9:
                print(f"✅ Priority attack SUCCESS!")
                return result
        
        print(f"   Priority attack completed: {len(priority_keys)} attempts")
        return self.best_result
    
    def ml_guided_attack(self, max_attempts: int = 5000) -> AttackResult:
        """Attack using ML-guided key generation"""
        print(f"\n🧠 Starting Pure NumPy ML-guided attack...")
        
        # Train on existing data
        if len(self.ml_system.training_data.key_features) > 20:
            print("   Training NumPy neural networks...")
            losses = self.ml_system.train_networks(epochs=5)
            print(f"   Training losses: {losses}")
        
        ciphertext_features = self.ml_system.extract_key_features(
            self.target_package.ciphertext.hex()[:self.config.key_hex_length]
        )
        
        successful_keys = []
        
        for i in range(max_attempts):
            if i % 500 == 0 and i > 0:
                print(f"   NumPy ML attempt {i}/{max_attempts}...")
                
                # Retrain periodically
                if len(self.ml_system.training_data.key_features) > 50:
                    losses = self.ml_system.train_networks(epochs=3)
            
            # Generate key using ML
            if random.random() < 0.7:  # 70% ML prediction
                key_candidate = self.ml_system.predict_key_candidate(ciphertext_features)
                method = "numpy_ml_prediction"
            else:  # 30% mutation of successful keys
                if successful_keys:
                    base_key = random.choice(successful_keys)
                    key_candidate = self.brute_force.generate_mutation(base_key)
                    method = "numpy_ml_mutation"
                else:
                    key_candidate = self.brute_force.generate_random_key()
                    method = "numpy_ml_random"
            
            result = self.attempt_decryption(key_candidate, method)
            
            if result.success and result.accuracy_score > 0.5:
                successful_keys.append(key_candidate)
                print(f"   🎯 Partial success: {result.accuracy_score:.1%} accuracy")
            
            if result.success and result.accuracy_score > 0.9:
                print(f"✅ NumPy ML attack SUCCESS!")
                return result
        
        print(f"   NumPy ML attack completed: {max_attempts} attempts")
        return self.best_result
    
    def brute_force_attack(self, max_attempts: int = 10000) -> AttackResult:
        """Pure brute force attack with intelligent key generation"""
        print(f"\n💥 Starting intelligent brute force attack...")
        
        for i in range(max_attempts):
            if i % 1000 == 0 and i > 0:
                print(f"   Brute force attempt {i}/{max_attempts}...")
            
            key_candidate = self.brute_force.generate_random_key()
            result = self.attempt_decryption(key_candidate, "brute_force")
            
            if result.success and result.accuracy_score > 0.9:
                print(f"✅ Brute force SUCCESS!")
                return result
        
        print(f"   Brute force completed: {max_attempts} attempts")
        return self.best_result
    
    def unlimited_attack(self, verbose: bool = False) -> AttackResult:
        """Unlimited attack that runs until success or interruption"""
        print(f"\n🚀 Starting UNLIMITED Pure NumPy AES attack...")
        print("⚠️  Press Ctrl+C to interrupt")
        
        phase = 1
        phase_attempts = 0
        phase_size = 2000
        
        try:
            while True:
                if verbose:
                    elapsed = time.time() - self.start_time
                    rate = self.attempts / max(1, elapsed)
                    print(f"\n📊 Phase {phase} - Attempt {self.attempts:,}")
                    print(f"   Rate: {rate:.1f} attempts/sec")
                    print(f"   Best accuracy: {self.best_result.accuracy_score:.1%}" if self.best_result else "   No success yet")
                
                # Adaptive strategy selection
                if phase <= 2:
                    # Start with priority patterns
                    key_candidate = self.brute_force.generate_priority_keys()
                    if key_candidate:
                        key = random.choice(key_candidate)
                        method = f"priority_phase_{phase}"
                    else:
                        key = self.brute_force.generate_random_key()
                        method = f"random_phase_{phase}"
                elif phase <= 5:
                    # ML-guided phase
                    if len(self.ml_system.training_data.key_features) > 100:
                        ciphertext_features = self.ml_system.extract_key_features(
                            self.target_package.ciphertext.hex()[:self.config.key_hex_length]
                        )
                        key = self.ml_system.predict_key_candidate(ciphertext_features)
                        method = f"numpy_ml_phase_{phase}"
                    else:
                        key = self.brute_force.generate_random_key()
                        method = f"training_phase_{phase}"
                else:
                    # Pure brute force with mutations
                    if self.best_result and self.best_result.accuracy_score > 0.1:
                        key = self.brute_force.generate_mutation(self.best_result.key_candidate)
                        method = f"mutation_phase_{phase}"
                    else:
                        key = self.brute_force.generate_random_key()
                        method = f"brute_phase_{phase}"
                
                result = self.attempt_decryption(key, method)
                phase_attempts += 1
                
                # Check for success
                if result.success and result.accuracy_score > 0.95:
                    print(f"\n🎉 UNLIMITED NUMPY ATTACK SUCCESS!")
                    print(f"   Phase: {phase}")
                    print(f"   Total attempts: {self.attempts:,}")
                    return result
                
                # Phase transition
                if phase_attempts >= phase_size:
                    phase += 1
                    phase_attempts = 0
                    
                    # Train ML systems between phases
                    if phase % 3 == 0 and len(self.ml_system.training_data.key_features) > 50:
                        if verbose:
                            print("   🧠 Training NumPy neural networks...")
                        losses = self.ml_system.train_networks(epochs=5)
                        
                    # Adaptive phase size
                    if phase > 10:
                        phase_size = min(5000, phase_size * 1.1)
        
        except KeyboardInterrupt:
            print(f"\n⚠️ Attack interrupted by user")
            print(f"   Completed {self.attempts:,} attempts in {phase} phases")
            return self.best_result
    
    def get_attack_summary(self) -> Dict[str, Any]:
        """Get comprehensive attack summary"""
        elapsed_time = time.time() - self.start_time
        
        return {
            'total_attempts': self.attempts,
            'elapsed_time': elapsed_time,
            'attempts_per_second': self.attempts / max(1, elapsed_time),
            'best_result': self.best_result,
            'success': self.best_result.success if self.best_result else False,
            'best_accuracy': self.best_result.accuracy_score if self.best_result else 0.0,
            'ml_examples': self.ml_system.training_data.total_examples,
            'target_analysis': self.ciphertext_analysis,
            'attack_methods_used': list(set([r.attack_method for r in self.attack_history]))
        }
    
    def save_progress(self):
        """Save attack progress and ML training data"""
        return self.ml_system.save_training_data()

def create_test_scenario(key_size: int = 256) -> Tuple[EncryptedPackage, str, str]:
    """Create a test scenario with known plaintext and key"""
    
    # Create test data
    test_messages = [
        "This is a secret message that needs to be encrypted using AES-CBC encryption.",
        "The quick brown fox jumps over the lazy dog. This is a test of AES encryption.",
        "Confidential data: User password is admin123, API key is secret_key_12345",
        "Meeting notes: Project deadline is December 31st. Budget approved for $50,000.",
        "System logs: Login successful for user@domain.com at 2024-01-15 14:30:22"
    ]
    
    plaintext = random.choice(test_messages)
    
    # Create encryption system
    config = AESConfig(key_size=key_size)
    constraints = KeyConstraints()
    crypto_engine = AESCryptographyEngine(config)
    
    # Generate a test key (following constraints)
    test_key = crypto_engine.generate_key(constraints)
    
    # Encrypt the message
    package = crypto_engine.encrypt_data(plaintext, test_key)
    
    return package, plaintext, test_key

def demonstrate_aes_cryptanalysis():
    """Demonstrate the Pure NumPy AES cryptanalysis system"""
    
    print("🔓 AES-CBC PURE NUMPY NEURAL NETWORK CRYPTANALYSIS")
    print("=" * 80)
    print("🧠 Advanced ML-powered attack using PURE NUMPY neural networks")
    print("💻 CPU-based neural networks with intelligent brute force")
    print("🎯 Features:")
    print("   • Pure NumPy neural network implementation")
    print("   • Key prediction with feedforward networks")
    print("   • Plaintext classification")
    print("   • Pattern learning")
    print("   • Intelligent brute force with constraints")
    print("   • ML-guided key generation")
    print("   • Training data persistence")
    print("   • Real AES-CBC-256/128 encryption")
    print("   • NO PYTORCH DEPENDENCY!")
    print()
    
    if not CRYPTO_AVAILABLE:
        print("❌ PyCryptodome required! Install with: pip install pycryptodome")
        return
    
    if not NUMPY_AVAILABLE:
        print("❌ NumPy required! Install with: pip install numpy")
        return
    
    print("🔧 Creating test scenario...")
    
    # Ask user for key size
    try:
        key_choice = input("Select key size (1=AES-128, 2=AES-256) [2]: ").strip()
        key_size = 128 if key_choice == "1" else 256
    except (EOFError, KeyboardInterrupt):
        key_size = 256
    
    # Create test scenario
    package, known_plaintext, actual_key = create_test_scenario(key_size)
    
    print(f"\n📋 Test Scenario Created:")
    print(f"   Algorithm: AES-CBC-{key_size}")
    print(f"   Plaintext: '{known_plaintext}'")
    print(f"   Actual key: {actual_key}")
    print(f"   Ciphertext length: {len(package.ciphertext)} bytes")
    print(f"   IV: {package.iv.hex()}")
    
    # Verify encryption works
    crypto_engine = AESCryptographyEngine(AESConfig(key_size=key_size))
    success, decrypted = crypto_engine.decrypt_data(package, actual_key)
    print(f"   Encryption test: {'✅ SUCCESS' if success and decrypted == known_plaintext else '❌ FAILED'}")
    
    if not success or decrypted != known_plaintext:
        print("❌ Encryption test failed - aborting demo")
        return
    
    # Initialize attacker
    print(f"\n🎯 Initializing Pure NumPy AES Cryptanalysis Attacker...")
    
    # Ask if we should provide known plaintext hint
    try:
        hint_choice = input("Provide known plaintext hint? (y/N): ").strip().lower()
        use_hint = hint_choice in ['y', 'yes']
    except (EOFError, KeyboardInterrupt):
        use_hint = False
    
    attacker = AESCryptanalysisAttacker(
        target_package=package,
        known_plaintext=known_plaintext if use_hint else ""
    )
    
    # Attack mode selection
    print(f"\n🚀 SELECT ATTACK MODE:")
    print("1. 🎯 Priority Pattern Attack (1K attempts)")
    print("2. 🧠 Pure NumPy ML-Guided Attack (5K attempts)")  
    print("3. 💥 Brute Force Attack (10K attempts)")
    print("4. 🔄 Unlimited Attack (runs until success)")
    print("5. 🎲 Demo Mode (all methods, quick)")
    
    try:
        choice = input("Enter choice (1-5) [4]: ").strip()
        if not choice:
            choice = "4"
    except (EOFError, KeyboardInterrupt):
        choice = "4"
    
    print(f"\n🚀 STARTING PURE NUMPY AES CRYPTANALYSIS ATTACK")
    print("=" * 60)
    print(f"🎯 Target: AES-CBC-{key_size}")
    print(f"📊 Known plaintext: {'Yes' if use_hint else 'No'}")
    print(f"🔑 Actual key (hidden): {actual_key}")
    print(f"🧠 Neural Networks: Pure NumPy implementation")
    
    start_time = time.time()
    
    try:
        if choice == "1":
            print(f"\n🎯 MODE 1: PRIORITY PATTERN ATTACK")
            result = attacker.priority_attack(max_attempts=1000)
            
        elif choice == "2":
            print(f"\n🧠 MODE 2: PURE NUMPY ML-GUIDED ATTACK")
            result = attacker.ml_guided_attack(max_attempts=5000)
            
        elif choice == "3":
            print(f"\n💥 MODE 3: BRUTE FORCE ATTACK")
            result = attacker.brute_force_attack(max_attempts=10000)
            
        elif choice == "4":
            print(f"\n🔄 MODE 4: UNLIMITED ATTACK")
            result = attacker.unlimited_attack(verbose=True)
            
        else:  # Demo mode
            print(f"\n🎲 MODE 5: DEMO MODE")
            print("Running all attack methods quickly...")
            
            # Quick priority attack
            print("\n1️⃣ Priority patterns...")
            result1 = attacker.priority_attack(max_attempts=100)
            
            # Quick ML attack
            print("\n2️⃣ Pure NumPy ML-guided...")
            result2 = attacker.ml_guided_attack(max_attempts=500)
            
            # Quick brute force
            print("\n3️⃣ Brute force...")
            result3 = attacker.brute_force_attack(max_attempts=1000)
            
            # Best result
            results = [r for r in [result1, result2, result3] if r]
            result = max(results, key=lambda x: x.accuracy_score) if results else None
    
    except KeyboardInterrupt:
        print(f"\n⚠️ Attack interrupted by user")
        result = attacker.best_result
    
    # Save progress
    attacker.save_progress()
    
    # Final analysis
    elapsed_time = time.time() - start_time
    summary = attacker.get_attack_summary()
    
    print(f"\n" + "🏆" * 20 + " PURE NUMPY RESULTS " + "🏆" * 20)
    
    if result and result.success and result.accuracy_score > 0.9:
        print(f"🎉 PURE NUMPY ATTACK SUCCESSFUL!")
        print(f"✅ Key found: {result.key_candidate}")
        print(f"✅ Matches actual: {result.key_candidate.lower() == actual_key.lower()}")
        print(f"📝 Decrypted text: '{result.plaintext_preview}'")
        security_status = "BROKEN BY PURE NUMPY"
    else:
        print(f"⚠️ Attack incomplete")
        if result:
            print(f"📈 Best accuracy: {result.accuracy_score:.1%}")
            print(f"🔑 Best key candidate: {result.key_candidate}")
            print(f"📝 Best decryption: '{result.plaintext_preview}'")
        security_status = f"PARTIALLY TESTED ({result.accuracy_score:.1%})" if result else "SECURE"
    
    print(f"\n📊 Attack Statistics:")
    print(f"   🔄 Total attempts: {summary['total_attempts']:,}")
    print(f"   ⏱️ Time taken: {elapsed_time:.1f} seconds")
    print(f"   ⚡ Rate: {summary['attempts_per_second']:.1f} attempts/sec")
    print(f"   🧠 ML examples collected: {summary['ml_examples']}")
    print(f"   🎯 Methods used: {', '.join(summary['attack_methods_used'])}")
    
    print(f"\n🧠 Pure NumPy Machine Learning Analysis:")
    print(f"   📚 Training examples: {attacker.ml_system.training_data.total_examples}")
    print(f"   🔄 Training sessions: {attacker.ml_system.training_data.session_count}")
    print(f"   📈 Neural networks: Pure NumPy implementation")
    print(f"     • Key predictor with multiple output heads")
    print(f"     • Plaintext classifier with sigmoid output")
    print(f"     • Pattern learner with feedforward architecture")
    print(f"   💾 Training data saved for future sessions")
    print(f"   ⚡ No PyTorch dependency required!")
    
    print(f"\n🛡️ Security Assessment:")
    print(f"   Algorithm: AES-CBC-{key_size}")
    print(f"   Status: {security_status}")
    print(f"   Method: Pure NumPy neural networks + intelligent brute force")
    
    if result and result.success:
        print(f"   Result: ❌ Encryption broken by Pure NumPy ML attack")
        print(f"   Implication: Key was vulnerable to pattern analysis")
    else:
        print(f"   Result: ✅ Encryption withstood attack")
        print(f"   Implication: Key appears resistant to current methods")
    
    print(f"\n💡 Pure NumPy Implementation Notes:")
    print(f"   🧠 Neural networks implemented from scratch")
    print(f"   📈 Adam optimizer with custom implementation")
    print(f"   🔢 Forward and backward propagation in pure NumPy")
    print(f"   ⚡ No external ML library dependencies")
    print(f"   📊 Demonstrates ML fundamentals without black boxes")
    
    print(f"\n🏁 Pure NumPy AES cryptanalysis demonstration complete!")
    
    return summary

if __name__ == "__main__":
    # Set random seeds for reproducible results
    random.seed(42)
    if NUMPY_AVAILABLE:
        np.random.seed(42)
    
    print("🔓 PURE NUMPY AES-CBC NEURAL NETWORK CRYPTANALYSIS")
    print("=" * 70)
    print("🧠 Machine learning attack using PURE NUMPY neural networks")
    print("💻 No PyTorch dependency - everything implemented from scratch")
    print("🎯 Educational cryptanalysis demonstration")
    print()
    print("⚡ Pure NumPy Features:")
    print("   🧠 Neural networks from scratch (Linear, ReLU, Sigmoid, Softmax)")
    print("   📈 Adam optimizer implementation")
    print("   🔢 Forward and backward propagation")
    print("   🔑 AES-CBC-256/128 encryption/decryption")
    print("   🎯 Intelligent key generation with constraints")
    print("   📊 Plaintext classification")
    print("   🔄 Pattern learning")
    print("   💾 Training data persistence")
    print("   🚀 Multiple attack strategies")
    print()
    print("📋 Requirements:")
    print("   pip install pycryptodome numpy")
    print("=" * 70)
    
    # Check dependencies
    missing_deps = []
    if not CRYPTO_AVAILABLE:
        missing_deps.append("pycryptodome")
    if not NUMPY_AVAILABLE:
        missing_deps.append("numpy")
    
    if missing_deps:
        print(f"❌ Missing dependencies: {', '.join(missing_deps)}")
        print(f"Install with: pip install {' '.join(missing_deps)}")
        print()
    else:
        print("✅ All dependencies available!")
        print()
    
    # Run demonstration
    try:
        results = demonstrate_aes_cryptanalysis()
        print(f"\n💡 Pure NumPy demonstration completed successfully!")
        
    except KeyboardInterrupt:
        print(f"\n⚠️ Demonstration interrupted by user")
        
    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()
        print(f"\n💡 Check that all dependencies are installed:")
        print(f"   pip install pycryptodome numpy")
    
    print(f"\n🎓 Educational Note:")
    print(f"This demonstrates ML-assisted cryptanalysis using pure NumPy.")
    print(f"Neural networks implemented from scratch show ML fundamentals.")
    print(f"AES with proper implementation and random keys remains secure.")
    print(f"Success depends on predictable key generation patterns.")
