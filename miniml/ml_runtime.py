"""
MiniML Runtime for Embedded Systems
----------------------
Runtime principal del framework.
Incluye implementaciones de algoritmos ML desde cero (sin dependencias externas).

Módulos incluidos:
 - MiniMatrixOps: Álgebra lineal básica.
 - Árboles: DecisionTree (Clasificación/Regresión), RandomForest.
 - Modelos Lineales: MiniLinearModel, MiniSVM.
 - Redes Neuronales: MiniNeuralNetwork (MLP).
 - Preprocesamiento: MiniScaler.
 - Lazy Learning: K-Nearest Neighbors (KNN).
 - Utilidades: Métricas y exportación a C.

MODIFICACIONES RECIENTES:
- Integración de check_dims en todos los métodos fit/predict.
- Almacenamiento de n_features_trained como metadata del modelo.
"""

from __future__ import annotations
from typing import List, Any, Optional, Tuple, Union, Dict
import random
import math
# Importamos utilidades de compatibilidad y validación
from .ml_compat import safe_compare_le, _flatten_tree_to_arrays, check_dims

# ---------------------------
# MiniMatrixOps (sin numpy)
# ---------------------------
class MiniMatrixOps:
    """Operaciones básicas de vectores/matrices sin dependencias externas."""

    @staticmethod
    def dot(a: List[float], b: List[float]) -> float:
        if len(a) != len(b):
            raise ValueError("Vectors must have same length for dot product")
        s = 0.0
        for i in range(len(a)):
            s += a[i] * b[i]
        return s

    @staticmethod
    def matvec(mat: List[List[float]], vec: List[float]) -> List[float]:
        return [MiniMatrixOps.dot(row, vec) for row in mat]

    @staticmethod
    def transpose(mat: List[List[float]]) -> List[List[float]]:
        if not mat:
            return []
        rows = len(mat)
        cols = len(mat[0])
        return [[mat[r][c] for r in range(rows)] for c in range(cols)]

    @staticmethod
    def outer(a: List[float], b: List[float]) -> List[List[float]]:
        return [[ai * bj for bj in b] for ai in a]

    @staticmethod
    def add_vec(a: List[float], b: List[float]) -> List[float]:
        if len(a) != len(b):
            raise ValueError("Vector length mismatch")
        return [a[i] + b[i] for i in range(len(a))]

    @staticmethod
    def scalar_mul_vec(s: float, v: List[float]) -> List[float]:
        return [s * x for x in v]

    @staticmethod
    def matrix_multiply(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
        """Multiplicación de matrices A (m x n) * B (n x p) -> (m x p)."""
        if not A or not B:
            return []
        m = len(A)
        n = len(A[0])
        if any(len(row) != n for row in A):
            raise ValueError("Invalid matrix A")
        if any(len(row) != len(B[0]) for row in B):
            pass 
        p = len(B[0])
        if any(len(row) != len(B[0]) for row in B):
            raise ValueError("Invalid matrix B")
        if len(B) != n:
            raise ValueError("Incompatible dimensions for matrix multiply")
        
        BT = MiniMatrixOps.transpose(B)
        result = []
        for i in range(m):
            row_res = []
            for j in range(p):
                row_res.append(MiniMatrixOps.dot(A[i], BT[j]))
            result.append(row_res)
        return result

# ---------------------------
# Utilities & Activations
# ---------------------------

def clip(value: float, min_val: float = -60.0, max_val: float = 60.0) -> float:
    """Protege activaciones (sigmoid overflow)."""
    if value < min_val:
        return min_val
    if value > max_val:
        return max_val
    return value

def sigmoid(x: float) -> float:
    x = clip(x)
    try:
        return 1.0 / (1.0 + math.exp(-x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0
    except Exception:
        return 0.5 

def sigmoid_derivative(output: float) -> float:
    return output * (1.0 - output)

def relu(x: float) -> float:
    return x if x > 0 else 0.0

def relu_derivative(x: float) -> float:
    return 1.0 if x > 0 else 0.0

def linear(x: float) -> float:
    return x

def linear_derivative(_: float) -> float:
    return 1.0

# ---------------------------
# Decision tree helpers (CART)
# ---------------------------
def split_dataset(dataset: List[List[Any]], feature_index: int, value: Any) -> Tuple[List[List[Any]], List[List[Any]]]:
    """Divide un conjunto de datos en dos grupos basándose en el valor de una característica."""
    left, right = [], []
    is_value_numeric = isinstance(value, (int, float))

    for row in dataset:
        try:
            if feature_index >= len(row):
                right.append(row)
                continue

            row_feature = row[feature_index]
            is_row_feature_numeric = isinstance(row_feature, (int, float))

            if is_value_numeric and is_row_feature_numeric:
                if row_feature <= value:
                    left.append(row)
                else:
                    right.append(row)
            else:
                right.append(row)
        except Exception:
            right.append(row)
            
    return left, right

def gini_index(groups, classes) -> float:
    n_instances = sum(len(g) for g in groups)
    if n_instances == 0:
        return 0.0
    gini = 0.0
    for group in groups:
        size = len(group)
        if size == 0:
            continue
        score = 0.0
        for class_val in classes:
            p = sum(1 for row in group if row[-1] == class_val) / size
            score += p * p
        gini += (1.0 - score) * (size / n_instances)
    return gini

def mse_index(groups) -> float:
    n_instances = sum(len(g) for g in groups)
    if n_instances == 0:
        return 0.0
    mse = 0.0
    for group in groups:
        size = len(group)
        if size == 0:
            continue
        mean = sum(row[-1] for row in group) / size
        sq_error = sum((row[-1] - mean) ** 2 for row in group)
        mse += sq_error * (size / n_instances)
    return mse

def to_terminal_class(group: List[List[Any]]):
    outcomes = {}
    for row in group:
        label = row[-1]
        outcomes[label] = outcomes.get(label, 0) + 1
    if not outcomes:
        return 0
    return max(outcomes.items(), key=lambda x: x[1])[0]

def to_terminal_reg(group: List[List[Any]]):
    if not group:
        return 0.0
    vals = [row[-1] for row in group]
    return sum(vals) / len(vals)

def get_split_class(dataset, n_features):
    class_values = list(set(row[-1] for row in dataset))
    b_index, b_value, b_score, b_groups = None, None, float('inf'), None
    
    if not dataset or not dataset[0]:
        return {'index': b_index, 'value': b_value, 'groups': b_groups}
        
    features = list(range(len(dataset[0]) - 1))
    if n_features is not None and n_features > 0:
        features = random.sample(features, max(1, min(len(features), n_features)))
        
    for index in features:
        values = set(row[index] for row in dataset)
        numeric_values = {v for v in values if isinstance(v, (int, float))}
        
        for value in numeric_values:
            groups = split_dataset(dataset, index, value)
            gini = gini_index(groups, class_values)
            
            if gini < b_score:
                b_index, b_value, b_score, b_groups = index, value, gini, groups
                
    return {'index': b_index, 'value': b_value, 'groups': b_groups}

def get_split_regression(dataset, n_features):
    b_index, b_value, b_score, b_groups = None, None, float('inf'), None

    if not dataset or not dataset[0]:
        return {'index': b_index, 'value': b_value, 'groups': b_groups}

    features = list(range(len(dataset[0]) - 1))
    if n_features is not None and n_features > 0:
        features = random.sample(features, max(1, min(len(features), n_features)))

    for index in features:
        values = set(row[index] for row in dataset)
        numeric_values = {v for v in values if isinstance(v, (int, float))}
        
        for value in numeric_values:
            groups = split_dataset(dataset, index, value)
            score = mse_index(groups)

            if score < b_score:
                b_index, b_value, b_score, b_groups = index, value, score, groups
                
    return {'index': b_index, 'value': b_value, 'groups': b_groups}

def build_tree_class(node, max_depth, min_size, n_features):
    groups = node.get('groups')
    if not groups or not isinstance(groups, (list, tuple)) or len(groups) != 2:
        node['left'] = to_terminal_class(node.get('groups') or [])
        node['right'] = node['left']
        node.pop('groups', None)
        return
    left, right = groups
    if not left or not right or max_depth <= 0:
        node['left'] = to_terminal_class(left)
        node['right'] = to_terminal_class(right)
        node.pop('groups', None)
        return
    left_child = get_split_class(left, n_features)
    node['left'] = left_child
    build_tree_class(left_child, max_depth - 1, min_size, n_features)
    right_child = get_split_class(right, n_features)
    node['right'] = right_child
    build_tree_class(right_child, max_depth - 1, min_size, n_features)
    node.pop('groups', None)

def build_tree_reg(node, max_depth, min_size, n_features):
    groups = node.get('groups')
    if not groups or not isinstance(groups, (list, tuple)) or len(groups) != 2:
        node['left'] = to_terminal_reg(node.get('groups') or [])
        node['right'] = node['left']
        node.pop('groups', None)
        return
    left, right = groups
    if not left or not right or max_depth <= 0:
        node['left'] = to_terminal_reg(left)
        node['right'] = to_terminal_reg(right)
        node.pop('groups', None)
        return
    left_child = get_split_regression(left, n_features)
    node['left'] = left_child
    build_tree_reg(left_child, max_depth - 1, min_size, n_features)
    right_child = get_split_regression(right, n_features)
    node['right'] = right_child
    build_tree_reg(right_child, max_depth - 1, min_size, n_features)
    node.pop('groups', None)

# ---------------------------
# Classifiers / Regressors
# ---------------------------

class DecisionTreeClassifier:
    def __init__(self, max_depth: int = 10, min_size: int = 1, n_features: Optional[int] = None):
        self.max_depth = max_depth
        self.min_size = min_size
        self.n_features = n_features
        self.root: Optional[Dict[str, Any]] = None
        self.n_features_trained: Optional[int] = None # Metadata dimensional

    def fit(self, dataset: List[List[Any]]):
        if not dataset or not isinstance(dataset, list):
            raise ValueError("Dataset invalid for fit()")
        
        # Guardar dimensiones de entrenamiento (dataset = features + target)
        self.n_features_trained = len(dataset[0]) - 1

        root = get_split_class(dataset, self.n_features)
        if not root.get('groups'):
            self.root = {'index': None, 'value': None, 'left': to_terminal_class(dataset), 'right': to_terminal_class(dataset)}
            return
        build_tree_class(root, self.max_depth, self.min_size, self.n_features)
        self.root = root

    def _predict_row(self, node, row):
        if not isinstance(node, dict):
            return node

        index = node.get('index')
        value = node.get('value')

        if index is None:
            if 'left' in node and not isinstance(node['left'], dict):
                return node['left']
            return node

        if isinstance(value, dict):
            return self._predict_row(value, row)

        try:
            row_feature = row[index]
            if not isinstance(row_feature, (int, float)):
                return self._predict_row(node.get('right'), row)
        except (IndexError, KeyError):
            return self._predict_row(node.get('right'), row)

        if safe_compare_le(row_feature, value):
            return self._predict_row(node.get('left'), row)
        else:
            return self._predict_row(node.get('right'), row)

    def predict(self, X: List[List[Any]]) -> List[Any]:
        if self.root is None:
            raise ValueError("Model not trained")
        
        # Validar dimensiones contra lo entrenado
        if self.n_features_trained is not None:
             check_dims(X, self.n_features_trained, "DecisionTree Predict")

        preds = []
        for row in X:
            preds.append(self._predict_row(self.root, row))
        return preds

    def to_arduino_code(self, fn_name: str = "predict_row") -> str:
        if self.root is None:
            return "// Error: Modelo no entrenado."
        
        flat = _flatten_tree_to_arrays(self.root)
        n_nodes = len(flat['feature_index'])
        
        # Helper para PROGMEM (Flash Memory)
        def progmem_arr(name, data, dtype):
            vals = ", ".join(map(str, data))
            return f"const {dtype} {name}[{len(data)}] PROGMEM = {{{vals}}};"

        code = [
            f"// --- MiniML Decision Tree ({n_nodes} nodes) ---",
            "// Optimized for AVR (Arduino): Uses PROGMEM to save SRAM.",
            "#include <avr/pgmspace.h>",
            "",
            progmem_arr(f"{fn_name}_idx", flat['feature_index'], "int16_t"),
            progmem_arr(f"{fn_name}_thr", flat['threshold'], "float"),
            progmem_arr(f"{fn_name}_left", flat['left_child'], "int16_t"),
            progmem_arr(f"{fn_name}_right", flat['right_child'], "int16_t"),
            progmem_arr(f"{fn_name}_val", flat['value'], "int16_t"),
            "",
            f"int {fn_name}(float *row) {{",
            "  int16_t n = 0;",
            "  while (1) {",
            f"    int16_t feat = (int16_t)pgm_read_word(&{fn_name}_idx[n]);",
            "    if (feat == -1) {",
            f"      return (int16_t)pgm_read_word(&{fn_name}_val[n]);",
            "    }",
            f"    float th = pgm_read_float(&{fn_name}_thr[n]);",
            "    if (row[feat] <= th) {",
            f"      n = (int16_t)pgm_read_word(&{fn_name}_left[n]);",
            "    } else {",
            f"      n = (int16_t)pgm_read_word(&{fn_name}_right[n]);",
            "    }",
            "  }",
            "}"
        ]
        return "\n".join(code)


class DecisionTreeRegressor(DecisionTreeClassifier):
    def fit(self, dataset: List[List[Any]]):
        if not dataset or not isinstance(dataset, list):
            raise ValueError("Dataset invalid for fit()")
        
        # Guardar dimensiones
        self.n_features_trained = len(dataset[0]) - 1

        root = get_split_regression(dataset, self.n_features)
        if not root.get('groups'):
            self.root = {'index': None, 'value': None, 'left': to_terminal_reg(dataset), 'right': to_terminal_reg(dataset)}
            return
        build_tree_reg(root, self.max_depth, self.min_size, self.n_features)
        self.root = root

    def predict(self, X: List[List[Any]]) -> List[float]:
        if self.root is None:
            raise ValueError("Model not trained")
            
        # Validar dimensiones
        if self.n_features_trained is not None:
             check_dims(X, self.n_features_trained, "DecisionTreeRegressor Predict")

        preds = []
        for row in X:
            p = self._predict_row(self.root, row)
            preds.append(float(p))
        return preds

    def to_arduino_code(self, fn_name: str = "predict_row") -> str:
        if self.root is None: return "// Error"
        flat = _flatten_tree_to_arrays(self.root)
        
        def progmem_arr(name, data, dtype):
            vals = ", ".join(map(str, data))
            return f"const {dtype} {name}[{len(data)}] PROGMEM = {{{vals}}};"

        code = [
            f"// --- MiniML Tree Regressor ---",
            "#include <avr/pgmspace.h>",
            progmem_arr(f"{fn_name}_idx", flat['feature_index'], "int16_t"),
            progmem_arr(f"{fn_name}_thr", flat['threshold'], "float"),
            progmem_arr(f"{fn_name}_left", flat['left_child'], "int16_t"),
            progmem_arr(f"{fn_name}_right", flat['right_child'], "int16_t"),
            progmem_arr(f"{fn_name}_val", flat['value'], "float"), # float value for regression
            "",
            f"float {fn_name}(float *row) {{",
            "  int16_t n = 0;",
            "  while (1) {",
            f"    int16_t feat = (int16_t)pgm_read_word(&{fn_name}_idx[n]);",
            "    if (feat == -1) {",
            f"      return pgm_read_float(&{fn_name}_val[n]);",
            "    }",
            f"    float th = pgm_read_float(&{fn_name}_thr[n]);",
            "    if (row[feat] <= th) {",
            f"      n = (int16_t)pgm_read_word(&{fn_name}_left[n]);",
            "    } else {",
            f"      n = (int16_t)pgm_read_word(&{fn_name}_right[n]);",
            "    }",
            "  }",
            "}"
        ]
        return "\n".join(code)

# ---------------------------
# Random Forest
# ---------------------------
class RandomForestClassifier:
    def __init__(self, n_trees: int = 5, max_depth: int = 10, min_size: int = 1,
                 sample_size: float = 1.0, n_features: Optional[int] = None, seed: Optional[int] = None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_size = min_size
        self.sample_size = sample_size
        self.n_features = n_features
        self.seed = seed
        self.trees: List[DecisionTreeClassifier] = []
        self.n_features_trained: Optional[int] = None # Metadata

    def _subsample(self, dataset):
        n_sample = max(1, int(len(dataset) * self.sample_size))
        return [random.choice(dataset) for _ in range(n_sample)]

    def fit(self, dataset: List[List[Any]]):
        if not dataset:
            raise ValueError("Dataset empty")
        
        # Guardar dimensiones
        self.n_features_trained = len(dataset[0]) - 1

        self.trees = []
        random.seed(self.seed)
        for i in range(self.n_trees):
            sample = self._subsample(dataset)
            tree = DecisionTreeClassifier(max_depth=self.max_depth, min_size=self.min_size, n_features=self.n_features)
            tree.fit(sample)
            self.trees.append(tree)

    def predict(self, X: List[List[Any]]) -> List[Any]:
        if not self.trees:
            raise ValueError("Not trained")
        
        # Validar dimensiones
        if self.n_features_trained is not None:
             check_dims(X, self.n_features_trained, "RandomForest Predict")

        votes = []
        for row in X:
            row_votes = [t._predict_row(t.root, row) for t in self.trees]
            agg = {}
            for v in row_votes:
                agg[v] = agg.get(v, 0) + 1
            votes.append(max(agg.items(), key=lambda x: x[1])[0])
        return votes

    def to_arduino_code(self, fn_name: str = "predict_rf") -> str:
        if not self.trees: return "// Error: Modelo no entrenado"

        code = [
            "// --- MiniML Random Forest Classifier (Optimized AVR) ---",
            "// All tree structures stored in PROGMEM to save SRAM.",
            "#include <avr/pgmspace.h>",
            ""
        ]

        # Helper para generar arrays PROGMEM
        def progmem_arr(name, data, dtype):
            if not data: return f"const {dtype} {name}[1] PROGMEM = {{0}};"
            vals = ", ".join(map(str, data))
            return f"const {dtype} {name}[{len(data)}] PROGMEM = {{{vals}}};"

        tree_functions = []

        for i, tree in enumerate(self.trees):
            if tree.root is None: continue
            
            flat = _flatten_tree_to_arrays(tree.root)
            prefix = f"{fn_name}_t{i}"
            
            # 1. Definir arrays globales en PROGMEM para este árbol
            code.append(f"// Tree {i}")
            code.append(progmem_arr(f"{prefix}_idx", flat['feature_index'], "int16_t"))
            code.append(progmem_arr(f"{prefix}_thr", flat['threshold'], "float"))
            code.append(progmem_arr(f"{prefix}_L", flat['left_child'], "int16_t"))
            code.append(progmem_arr(f"{prefix}_R", flat['right_child'], "int16_t"))
            code.append(progmem_arr(f"{prefix}_val", flat['value'], "int16_t"))
            
            # 2. Función de inferencia específica para este árbol
            t_func_name = f"{prefix}_predict"
            tree_functions.append(t_func_name)
            
            func_code = [
                f"int16_t {t_func_name}(float *row) {{",
                "  int16_t n = 0;",
                "  while (1) {",
                f"    int16_t feat = (int16_t)pgm_read_word(&{prefix}_idx[n]);",
                "    if (feat == -1) {",
                f"      return (int16_t)pgm_read_word(&{prefix}_val[n]);",
                "    }",
                f"    float th = pgm_read_float(&{prefix}_thr[n]);",
                "    if (row[feat] <= th) {",
                f"      n = (int16_t)pgm_read_word(&{prefix}_L[n]);",
                "    } else {",
                f"      n = (int16_t)pgm_read_word(&{prefix}_R[n]);",
                "    }",
                "  }",
                "}",
                ""
            ]
            code.extend(func_code)

        # 3. Función de votación
        code.append("// Majority Voting Helper")
        code.append(f"int {fn_name}(float *row) {{")
        code.append(f"  int votes[{len(tree_functions)}];")
        
        for i, t_func in enumerate(tree_functions):
            code.append(f"  votes[{i}] = {t_func}(row);")
            
        code.append(f"  // Voting Logic (O(N^2) simple impl for small N)")
        code.append(f"  int max_count = 0;")
        code.append(f"  int best_vote = votes[0];")
        code.append(f"  for (int i=0; i<{len(tree_functions)}; i++) {{")
        code.append(f"    int c = 0;")
        code.append(f"    for (int j=0; j<{len(tree_functions)}; j++) {{")
        code.append(f"      if (votes[j] == votes[i]) c++;")
        code.append(f"    }}")
        code.append(f"    if (c > max_count) {{ max_count = c; best_vote = votes[i]; }}")
        code.append(f"  }}")
        code.append(f"  return best_vote;")
        code.append("}")

        return "\n".join(code)

class RandomForestRegressor(RandomForestClassifier):
    def fit(self, dataset: List[List[Any]]):
        if not dataset:
            raise ValueError("Dataset empty")
            
        # Guardar dimensiones
        self.n_features_trained = len(dataset[0]) - 1

        self.trees = []
        random.seed(self.seed)
        for i in range(self.n_trees):
            sample = self._subsample(dataset)
            tree = DecisionTreeRegressor(max_depth=self.max_depth, min_size=self.min_size, n_features=self.n_features)
            tree.fit(sample)
            self.trees.append(tree)

    def predict(self, X: List[List[Any]]) -> List[float]:
        if not self.trees:
            raise ValueError("Not trained")
        
        # Validar dimensiones
        if self.n_features_trained is not None:
             check_dims(X, self.n_features_trained, "RandomForestReg Predict")

        preds = []
        for row in X:
            row_preds = [t._predict_row(t.root, row) for t in self.trees]
            avg = sum(float(p) for p in row_preds) / len(row_preds)
            preds.append(avg)
        return preds

    def to_arduino_code(self, fn_name: str = "predict_rf_reg") -> str:
        if not self.trees: return "// Error: Modelo no entrenado"

        code = [
            "// --- MiniML RF Regressor (Optimized AVR) ---",
            "#include <avr/pgmspace.h>",
            ""
        ]

        def progmem_arr(name, data, dtype):
            if not data: return f"const {dtype} {name}[1] PROGMEM = {{0}};"
            vals = ", ".join(map(str, data))
            return f"const {dtype} {name}[{len(data)}] PROGMEM = {{{vals}}};"

        tree_functions = []

        for i, tree in enumerate(self.trees):
            if tree.root is None: continue
            flat = _flatten_tree_to_arrays(tree.root)
            prefix = f"{fn_name}_t{i}"
            
            code.append(f"// Tree {i}")
            code.append(progmem_arr(f"{prefix}_idx", flat['feature_index'], "int16_t"))
            code.append(progmem_arr(f"{prefix}_thr", flat['threshold'], "float"))
            code.append(progmem_arr(f"{prefix}_L", flat['left_child'], "int16_t"))
            code.append(progmem_arr(f"{prefix}_R", flat['right_child'], "int16_t"))
            # Nota: para regresión, value es float
            code.append(progmem_arr(f"{prefix}_val", flat['value'], "float"))
            
            t_func_name = f"{prefix}_predict"
            tree_functions.append(t_func_name)
            
            func_code = [
                f"float {t_func_name}(float *row) {{",
                "  int16_t n = 0;",
                "  while (1) {",
                f"    int16_t feat = (int16_t)pgm_read_word(&{prefix}_idx[n]);",
                "    if (feat == -1) {",
                f"      return pgm_read_float(&{prefix}_val[n]);",
                "    }",
                f"    float th = pgm_read_float(&{prefix}_thr[n]);",
                "    if (row[feat] <= th) {",
                f"      n = (int16_t)pgm_read_word(&{prefix}_L[n]);",
                "    } else {",
                f"      n = (int16_t)pgm_read_word(&{prefix}_R[n]);",
                "    }",
                "  }",
                "}",
                ""
            ]
            code.extend(func_code)

        # Average logic
        code.append(f"float {fn_name}(float *row) {{")
        code.append(f"  float sum = 0.0;")
        for t_func in tree_functions:
            code.append(f"  sum += {t_func}(row);")
        code.append(f"  return sum / {len(tree_functions)}.0;")
        code.append("}")

        return "\n".join(code)

# ---------------------------
# Mini Linear Model
# ---------------------------
class MiniLinearModel:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = float(learning_rate)
        self.epochs = int(epochs)
        self.weights = None
        self.n_features_trained: Optional[int] = None # Metadata

    def _unpack(self, dataset):
        X = [row[:-1] for row in dataset]
        y = [row[-1] for row in dataset]
        return X, y

    def fit(self, dataset):
        X, y = self._unpack(dataset)
        if not X:
            raise ValueError("Empty dataset")
        
        # Guardar dimensiones
        self.n_features_trained = len(X[0])

        n_samples = len(X)
        n_features = len(X[0])
        self.weights = [0.0] * n_features + [0.0]
        for epoch in range(self.epochs):
            grads = [0.0] * (n_features + 1)
            for xi, yi in zip(X, y):
                pred = sum(w * xv for w, xv in zip(self.weights[:-1], xi)) + self.weights[-1]
                err = pred - yi
                for j in range(n_features):
                    grads[j] += (2.0 / n_samples) * err * xi[j]
                grads[-1] += (2.0 / n_samples) * err
            for j in range(n_features + 1):
                self.weights[j] -= self.learning_rate * grads[j]

    def predict(self, X_list):
        if self.weights is None:
            raise ValueError("Model not trained")
        
        # Validar dimensiones
        if self.n_features_trained is not None:
             check_dims(X_list, self.n_features_trained, "MiniLinear Predict")

        preds = []
        for xi in X_list:
            if not isinstance(xi, (list, tuple)):
                xi = [xi]
            pred = sum(w * xv for w, xv in zip(self.weights[:-1], xi)) + self.weights[-1]
            preds.append(pred)
        return preds

    def to_arduino_code(self, fn_name="predict_lin"):
        if not self.weights: return "// Error: Modelo no entrenado"
        w = self.weights
        n_w = len(w)
        
        # PROGMEM: Guardamos los pesos en Flash
        code = [
            f"// --- MiniLinearModel Optimized (AVR) ---",
            "#include <avr/pgmspace.h>",
            f"const float {fn_name}_weights[{n_w}] PROGMEM = {{{', '.join(map(str, w))}}};",
            "",
            f"float {fn_name}(float *row) {{",
            "  float s = 0.0;",
            f"  // Producto punto leyendo desde Flash",
            f"  for (int i = 0; i < {n_w - 1}; i++) {{",
            f"    float w = pgm_read_float(&{fn_name}_weights[i]);",
            "    s += w * row[i];",
            "  }",
            f"  // Bias (ultimo peso)",
            f"  s += pgm_read_float(&{fn_name}_weights[{n_w - 1}]);",
            "  return s;",
            "}"
        ]
        return "\n".join(code)

# ---------------------------
# Mini SVM (simple linear)
# ---------------------------
class MiniSVM:
    def __init__(self, learning_rate=0.01, lambda_param=0.01, n_iters=1000):
        self.learning_rate = float(learning_rate)
        self.lambda_param = float(lambda_param)
        self.n_iters = int(n_iters)
        self.weights = None
        self.n_features_trained: Optional[int] = None # Metadata

    def fit(self, dataset):
        X = [row[:-1] for row in dataset]
        y = [row[-1] for row in dataset]
        if not X:
            raise ValueError("Empty dataset")
        
        # Guardar dimensiones
        self.n_features_trained = len(X[0])
        n_features = len(X[0])

        self.weights = [0.0] * (n_features + 1)
        for it in range(self.n_iters):
            for xi, yi in zip(X, y):
                if not isinstance(yi, (int, float)):
                    raise ValueError("Labels must be numeric")
                yi = 1 if yi > 0 else -1
                wx = sum(w * xv for w, xv in zip(self.weights[:-1], xi)) + self.weights[-1]
                if yi * wx < 1:
                    for j in range(n_features):
                        self.weights[j] = (1 - self.learning_rate * self.lambda_param) * self.weights[j] + self.learning_rate * yi * xi[j]
                    self.weights[-1] = (1 - self.learning_rate * self.lambda_param) * self.weights[-1] + self.learning_rate * yi
                else:
                    for j in range(n_features + 1):
                        self.weights[j] = (1 - self.learning_rate * self.lambda_param) * self.weights[j]

    def predict(self, X_list):
        if self.weights is None:
            raise ValueError("Model not trained")
        
        # Validar dimensiones
        if self.n_features_trained is not None:
             check_dims(X_list, self.n_features_trained, "MiniSVM Predict")

        out = []
        for xi in X_list:
            if not isinstance(xi, (list, tuple)):
                xi = [xi]
            s = sum(w * xv for w, xv in zip(self.weights[:-1], xi)) + self.weights[-1]
            out.append(1 if s >= 0 else -1)
        return out

    def to_arduino_code(self, fn_name="predict_svm"):
        if not self.weights: return "// Error: Modelo no entrenado"
        w = self.weights
        n_w = len(w)

        code = [
            f"// --- MiniSVM Optimized (AVR) ---",
            "#include <avr/pgmspace.h>",
            f"const float {fn_name}_weights[{n_w}] PROGMEM = {{{', '.join(map(str, w))}}};",
            "",
            f"int {fn_name}(float *row) {{",
            "  float s = 0.0;",
            f"  for (int i = 0; i < {n_w - 1}; i++) {{",
            f"    float w = pgm_read_float(&{fn_name}_weights[i]);",
            "    s += w * row[i];",
            "  }",
            f"  s += pgm_read_float(&{fn_name}_weights[{n_w - 1}]);",
            "  return (s >= 0.0) ? 1 : -1;",
            "}"
        ]
        return "\n".join(code)

# ---------------------------
# MiniNeuralNetwork
# ---------------------------
class MiniNeuralNetwork:
    def __init__(self, n_inputs, n_hidden, n_outputs, learning_rate=0.1, epochs=1000, seed=None):
        self.n_inputs = int(n_inputs)
        self.n_hidden = int(n_hidden)
        self.n_outputs = int(n_outputs)
        self.learning_rate = float(learning_rate)
        self.epochs = int(epochs)
        self.quantized = False
        self.act_scales = {}
        self.hidden_activation = "sigmoid"
        self.output_activation = "sigmoid"
        
        if seed is not None:
            random.seed(seed)
        
        def rand_matrix(rows, cols):
            return [[(random.random() - 0.5) * 0.2 for _ in range(cols)] for _ in range(rows)]
        
        limit1 = math.sqrt(6 / (self.n_inputs + self.n_hidden))
        self.W1 = [[random.uniform(-limit1, limit1) for _ in range(self.n_inputs)] for _ in range(self.n_hidden)]
        self.B1 = [[0.0] for _ in range(self.n_hidden)]
        
        limit2 = math.sqrt(6 / (self.n_hidden + self.n_outputs))
        self.W2 = [[random.uniform(-limit2, limit2) for _ in range(self.n_hidden)] for _ in range(self.n_outputs)]
        self.B2 = [[0.0] for _ in range(self.n_outputs)]

        # Atributos para cuantificación (se llenan en quantize)
        self.q_W1 = []
        self.i32_B1 = []
        self.requant_mult1 = []
        self.s_W1_list = []
        
        self.q_W2 = []
        self.i32_B2 = []
        self.requant_mult2 = []
        self.s_W2_list = []

    def clip(self, value, min_val=-60.0, max_val=60.0):
        if value < min_val: return min_val
        if value > max_val: return max_val
        return value

    def sigmoid(self, x):
        # Protección contra overflow en exp
        if x > 60: return 1.0
        if x < -60: return 0.0
        return 1.0 / (1.0 + math.exp(-x))

    def sigmoid_deriv(self, out_val):
        return out_val * (1.0 - out_val)

    def relu(self, x):
        return x if x > 0 else 0.0

    def relu_derivative(self, x):
        return 1.0 if x > 0 else 0.0

    def linear(self, x):
        return x
    
    def linear_derivative(self, x):
        return 1.0

    def _activate(self, x, act):
        if act == 'sigmoid': return self.sigmoid(x)
        if act == 'relu': return self.relu(x)
        if act == 'linear': return self.linear(x)
        return self.sigmoid(x)

    def _act_derivative(self, out_val, act, pre_x=None):
        # Nota: Para ReLU, la derivada idealmente usa el valor pre-activación (pre_x),
        # pero a menudo se aproxima usando el output.
        if act == 'sigmoid': return self.sigmoid_deriv(out_val)
        if act == 'relu': return self.relu_derivative(out_val) 
        if act == 'linear': return self.linear_derivative(out_val)
        return self.sigmoid_deriv(out_val)

    def _forward(self, x_row):
        z1, a1 = [], []
        for i in range(self.n_hidden):
            s = sum(self.W1[i][j] * x_row[j] for j in range(self.n_inputs)) + self.B1[i][0]
            si = self._activate(s, getattr(self, "hidden_activation", "sigmoid"))
            z1.append(s)
            a1.append(si)

        z2, a2 = [], []
        for k in range(self.n_outputs):
            s = sum(self.W2[k][i] * a1[i] for i in range(self.n_hidden)) + self.B2[k][0]
            si = self._activate(s, getattr(self, "output_activation", "sigmoid"))
            z2.append(s)
            a2.append(si)

        return a1, a2

    def fit(self, dataset: List[List[Any]]):
        """
        Entrena la red neuronal. 
        Ahora acepta un 'dataset' unificado [features + target] para compatibilidad con ml_manager.
        """
        if not dataset:
            raise ValueError("Empty dataset")
        
        # Desempacar dataset (Standardization con el resto del framework)
        X = [row[:-1] for row in dataset]
        y = [row[-1] for row in dataset]

        # Validar dimensiones
        check_dims(X, self.n_inputs, "MiniNeuralNetwork Fit")
        
        # Formatear targets (manejo de escalares a listas)
        y_formatted = []
        for yi in y:
            if isinstance(yi, (list, tuple)):
                y_formatted.append([float(v) for v in yi])
            else:
                y_formatted.append([float(yi)])
            
        # Loop de entrenamiento
        for epoch in range(self.epochs):
            for xi, yi in zip(X, y_formatted):
                a1, a2 = self._forward(xi)
                
                delta2 = [0.0] * self.n_outputs
                for k in range(self.n_outputs):
                    err = a2[k] - yi[k]
                    delta2[k] = err * self._act_derivative(a2[k], self.output_activation)
                
                delta1 = [0.0] * self.n_hidden
                for i in range(self.n_hidden):
                    s = 0.0
                    for k in range(self.n_outputs):
                        s += self.W2[k][i] * delta2[k]
                    delta1[i] = s * self._act_derivative(a1[i], self.hidden_activation)
                
                for k in range(self.n_outputs):
                    for i in range(self.n_hidden):
                        self.W2[k][i] -= self.learning_rate * delta2[k] * a1[i]
                    self.B2[k][0] -= self.learning_rate * delta2[k]
                
                for i in range(self.n_hidden):
                    for j in range(self.n_inputs):
                        self.W1[i][j] -= self.learning_rate * delta1[i] * xi[j]
                    self.B1[i][0] -= self.learning_rate * delta1[i]

        self.calibrate(dataset)

    def predict(self, X_list):
        # Validar dimensiones
        check_dims(X_list, self.n_inputs, "MiniNeuralNetwork Predict")

        preds = []
        for xi in X_list:
            _, a2 = self._forward(xi)
            if self.n_outputs == 1:
                preds.append([a2[0]])
            else:
                preds.append(a2[:])
        return preds

    def calibrate(self, dataset: List[List[float]]):
        """
        Calcula rangos de activación (min/max) para Input, Hidden y Output.
        Esencial para cuantificación int8 (Post-Training Quantization).
        """
        # Validación básica para no calibrar con basura
        if not dataset: return
        
        # Detectar si dataset tiene target o solo features
        sample_row = dataset[0]
        # Si la fila es más larga que las entradas esperadas, asumimos que el último es target
        if len(sample_row) > self.n_inputs:
            X = [row[:-1] for row in dataset]
        else:
            X = dataset

        max_in, max_hidden, max_out = 0.0, 0.0, 0.0

        for x in X:
            # Input
            local_max_in = max(abs(xi) for xi in x)
            if local_max_in > max_in: max_in = local_max_in

            # Hidden
            a1_vals = []
            for i in range(self.n_hidden):
                s = sum(self.W1[i][j] * x[j] for j in range(self.n_inputs)) + self.B1[i][0]
                val = self._activate(s, self.hidden_activation)
                a1_vals.append(val)
            local_max_hidden = max(abs(v) for v in a1_vals)
            if local_max_hidden > max_hidden: max_hidden = local_max_hidden

            # Output
            a2_vals = []
            for k in range(self.n_outputs):
                s = sum(self.W2[k][i] * a1_vals[i] for i in range(self.n_hidden)) + self.B2[k][0]
                val = self._activate(s, self.output_activation)
                a2_vals.append(val)
            local_max_out = max(abs(v) for v in a2_vals)
            if local_max_out > max_out: max_out = local_max_out

        if max_in < 1e-9: max_in = 1.0
        if max_hidden < 1e-9: max_hidden = 1.0
        if max_out < 1e-9: max_out = 1.0

        self.act_scales = {
            'input': max_in / 127.0,
            'hidden': max_hidden / 127.0,
            'output': max_out / 127.0
        }

    def quantize(self, per_channel: bool = True):
        if self.quantized: return
        # Ahora act_scales debería existir siempre tras un fit()
        if not self.act_scales:
            raise RuntimeError("Model not calibrated. Run calibrate() before quantize().")

        s_in = self.act_scales['input']
        s_hidden = self.act_scales['hidden']
        s_out = self.act_scales['output']

        def quantize_layer(weights, biases, input_scale, output_scale):
            q_w_mat = []
            q_b_vec = []
            mult_vec = []
            scale_w_vec = []
            for i, row in enumerate(weights):
                max_w = max(abs(w) for w in row)
                if max_w < 1e-9: max_w = 1e-9
                s_w = max_w / 127.0
                scale_w_vec.append(s_w)
                
                q_row = [int(round(w / s_w)) for w in row]
                q_row = [max(-127, min(127, x)) for x in q_row]
                q_w_mat.append(q_row)
                
                effective_scale = input_scale * s_w
                b_val = biases[i][0]
                b_int = int(round(b_val / effective_scale))
                b_int = max(-2147483648, min(2147483647, b_int))
                q_b_vec.append(b_int)
                
                if output_scale < 1e-12: m = 0.0
                else: m = effective_scale / output_scale
                mult_vec.append(m)
            return q_w_mat, q_b_vec, mult_vec, scale_w_vec

        self.q_W1, self.i32_B1, self.requant_mult1, self.s_W1_list = quantize_layer(self.W1, self.B1, s_in, s_hidden)
        self.q_W2, self.i32_B2, self.requant_mult2, self.s_W2_list = quantize_layer(self.W2, self.B2, s_hidden, s_out)
        self.quantized = True

    def to_arduino_code(self, fn_name="nn_predict"):
        """
        Genera código C optimizado para AVR (Arduino Uno/Nano).
        Modo Híbrido: Pesos int8 en PROGMEM (Flash), Cálculo en Float.
        Ahorra mucha SRAM, ideal para 8-bit.
        """
        # Asegurar cuantificación
        if not self.quantized:
            print("[MiniML] Warning: Exporting unquantized model to AVR.")
            # Si no hay escalas, inventamos unas seguras (Dummy Calibration) para no romper el flujo
            if not self.act_scales:
                print("[MiniML] Auto-calibrating with default range [-1, 1] for export safety...")
                self.act_scales = {'input': 1.0/127.0, 'hidden': 1.0/127.0, 'output': 1.0/127.0}
            
            self.quantize(per_channel=True)

        lines = [
            f"// --- MiniML MLP (Optimized for 8-bit AVR) ---",
            "// Strategy: Hybrid (Int8 Storage in Flash, Float Compute)",
            "// Saves SRAM using PROGMEM.",
            "#include <avr/pgmspace.h>",
            "#include <math.h>"
        ]
        
        def to_c_matrix(matrix): 
            return '{' + ','.join('{' + ','.join(map(str, r)) + '}' for r in matrix) + '}'
        
        def to_c_array(arr):
            return '{' + ', '.join(map(str, arr)) + '}'

        # Layer 1 Generation
        # Pesos (int8) en PROGMEM
        lines.append(f"const int8_t {fn_name}_W1[{self.n_hidden}][{self.n_inputs}] PROGMEM = {to_c_matrix(self.q_W1)};")
        
        # Escalas (float) en PROGMEM - Usamos s_W1_list (La nueva API)
        lines.append(f"const float {fn_name}_sW1[{self.n_hidden}] PROGMEM = {to_c_array(self.s_W1_list)};")
        
        # Bias (float original) en PROGMEM
        # Nota: En modo híbrido AVR, es más rápido leer el float original que reconstruirlo de int32
        bias1_floats = [b[0] for b in self.B1]
        lines.append(f"const float {fn_name}_B1[{self.n_hidden}] PROGMEM = {to_c_array(bias1_floats)};")

        # Layer 2 Generation
        lines.append(f"const int8_t {fn_name}_W2[{self.n_outputs}][{self.n_hidden}] PROGMEM = {to_c_matrix(self.q_W2)};")
        lines.append(f"const float {fn_name}_sW2[{self.n_outputs}] PROGMEM = {to_c_array(self.s_W2_list)};")
        
        bias2_floats = [b[0] for b in self.B2]
        lines.append(f"const float {fn_name}_B2[{self.n_outputs}] PROGMEM = {to_c_array(bias2_floats)};")

        # Inference Function
        lines.append(f"void {fn_name}(float *row, float *out) {{")
        lines.append(f"  float a1[{self.n_hidden}];")
        
        # Loop Capa 1
        lines.append(f"  for(int i=0; i<{self.n_hidden}; i++) {{")
        lines.append("    float sum = 0.0;")
        lines.append(f"    float s = pgm_read_float(&{fn_name}_sW1[i]); // Read Scale")
        
        lines.append(f"    for(int j=0; j<{self.n_inputs}; j++) {{")
        lines.append(f"      int8_t w = (int8_t)pgm_read_byte(&{fn_name}_W1[i][j]); // Read Weight")
        lines.append(f"      sum += (float)w * s * row[j]; // Dequantize on-the-fly")
        lines.append("    }")
        lines.append(f"    sum += pgm_read_float(&{fn_name}_B1[i]); // Add Bias")
        
        # Activación L1
        if self.hidden_activation == 'relu':
            lines.append("    a1[i] = (sum > 0) ? sum : 0.0;")
        else: # Sigmoid
            lines.append("    if(sum>10) a1[i]=1.0; else if(sum<-10) a1[i]=0.0; else a1[i]=1.0/(1.0+exp(-sum));")
        lines.append("  }")

        # Loop Capa 2
        lines.append(f"  for(int k=0; k<{self.n_outputs}; k++) {{")
        lines.append("    float sum = 0.0;")
        lines.append(f"    float s = pgm_read_float(&{fn_name}_sW2[k]);")

        lines.append(f"    for(int i=0; i<{self.n_hidden}; i++) {{")
        lines.append(f"      int8_t w = (int8_t)pgm_read_byte(&{fn_name}_W2[k][i]);")
        lines.append(f"      sum += (float)w * s * a1[i];")
        lines.append("    }")
        lines.append(f"    sum += pgm_read_float(&{fn_name}_B2[k]);")

        # Activación L2
        if self.output_activation == 'relu':
            lines.append("    out[k] = (sum > 0) ? sum : 0.0;")
        else:
             lines.append("    if(sum>10) out[k]=1.0; else if(sum<-10) out[k]=0.0; else out[k]=1.0/(1.0+exp(-sum));")
        lines.append("  }")
        lines.append("}")

        return "\n".join(lines)

# ---------------------------
# Preprocessing
# ---------------------------
class MiniScaler:
    """
    Escalador para normalizar datos (MinMax o Standard).
    Crucial para KNN, SVM y Redes Neuronales.
    """
    def __init__(self, method='minmax', feature_range=(0, 1)):
        self.method = method
        self.feature_range = feature_range
        self.params = [] 
        self.n_features_trained: Optional[int] = None # Metadata

    def fit(self, dataset: List[List[float]]):
        if not dataset:
            return
        
        # Guardar dimensiones
        self.n_features_trained = len(dataset[0])
        n_features = len(dataset[0])
        
        self.params = []
        
        cols = [[row[i] for row in dataset] for i in range(n_features)]
        
        for col in cols:
            stats = {}
            if self.method == 'minmax':
                stats['min'] = min(col)
                stats['max'] = max(col)
                stats['denom'] = stats['max'] - stats['min']
                if stats['denom'] == 0: stats['denom'] = 1.0
            elif self.method == 'standard':
                mean = sum(col) / len(col)
                variance = sum((x - mean) ** 2 for x in col) / len(col)
                stats['mean'] = mean
                stats['std'] = math.sqrt(variance)
                if stats['std'] == 0: stats['std'] = 1.0
            self.params.append(stats)

    def transform(self, row: List[float]) -> List[float]:
        if not self.params:
            return row
            
        # Validación de dimensiones si está disponible
        if isinstance(row, list) and self.n_features_trained:
             if len(row) != self.n_features_trained:
                 raise ValueError(f"Dimensión incorrecta. Esperado {self.n_features_trained}, recibido {len(row)}")

        new_row = []
        for i, val in enumerate(row):
            if i >= len(self.params): 
                new_row.append(val)
                continue
            
            p = self.params[i]
            if self.method == 'minmax':
                norm = (val - p['min']) / p['denom']
                scaled = norm * (self.feature_range[1] - self.feature_range[0]) + self.feature_range[0]
                new_row.append(scaled)
            elif self.method == 'standard':
                scaled = (val - p['mean']) / p['std']
                new_row.append(scaled)
        return new_row

    def to_arduino_code(self, fn_name="preprocess_row"):
        if not self.params: return "// Error: Scaler not fitted"
        n = len(self.params)
        
        # Helper para arrays en Flash
        def flash_arr(name, data):
            return f"const float {name}[{n}] PROGMEM = {{{', '.join(map(str, data))}}};"

        code = [f"// --- MiniScaler ({self.method}) ---", "#include <avr/pgmspace.h>"]
        
        if self.method == 'minmax':
            mins = [p['min'] for p in self.params]
            denoms = [p['denom'] for p in self.params]
            code.append(flash_arr(f"{fn_name}_min", mins))
            code.append(flash_arr(f"{fn_name}_den", denoms))
            
            code.append(f"void {fn_name}(float *row) {{")
            code.append(f"  for(int i=0; i<{n}; i++) {{")
            code.append(f"    float mn = pgm_read_float(&{fn_name}_min[i]);")
            code.append(f"    float dn = pgm_read_float(&{fn_name}_den[i]);")
            code.append(f"    row[i] = (row[i] - mn) / dn;")
            if self.feature_range != (0, 1):
                r_min, r_max = self.feature_range
                scale = r_max - r_min
                code.append(f"    row[i] = row[i] * {scale} + {r_min};")
            code.append("  }")  # Cerrar for loop (una llave)
            code.append("}")    # Cerrar función
            
        elif self.method == 'standard':
            means = [p['mean'] for p in self.params]
            stds = [p['std'] for p in self.params]
            code.append(flash_arr(f"{fn_name}_mean", means))
            code.append(flash_arr(f"{fn_name}_std", stds))
            
            code.append(f"void {fn_name}(float *row) {{")
            code.append(f"  for(int i=0; i<{n}; i++) {{")
            code.append(f"    float mu = pgm_read_float(&{fn_name}_mean[i]);")
            code.append(f"    float sig = pgm_read_float(&{fn_name}_std[i]);")
            code.append(f"    row[i] = (row[i] - mu) / sig;")
            code.append("  }")  # Cerrar for loop (una llave)
            code.append("}")    # Cerrar función

        return "\n".join(code)

# ---------------------------
# K-Nearest Neighbors
# ---------------------------
class KNearestNeighbors:
    def __init__(self, k=3, task='classification', **kwargs):
        self.k = k
        self.task = kwargs.get('task-type', task)
        self.X_train = []
        self.y_train = []
        self.n_features_trained: Optional[int] = None # Metadata

    def fit(self, dataset):
        if not dataset:
            raise ValueError("Empty dataset")
        
        # Guardar dimensiones
        self.n_features_trained = len(dataset[0]) - 1

        self.X_train = [row[:-1] for row in dataset]
        self.y_train = [row[-1] for row in dataset]

    def _euclidean_dist(self, row1, row2):
        dist = 0.0
        for i in range(len(row1)):
            dist += (row1[i] - row2[i])**2
        return math.sqrt(dist)

    def predict(self, X_list):
        if not self.X_train:
            raise ValueError("Not trained")
        
        # Validar dimensiones
        if self.n_features_trained is not None:
             check_dims(X_list, self.n_features_trained, "KNN Predict")

        preds = []
        for row in X_list:
            distances = []
            for i, train_row in enumerate(self.X_train):
                d = self._euclidean_dist(row, train_row)
                distances.append((d, self.y_train[i]))
            
            distances.sort(key=lambda x: x[0])
            neighbors = distances[:self.k]
            
            if self.task == 'regression':
                val = sum(n[1] for n in neighbors) / self.k
                preds.append(val)
            else:
                votes = {}
                for _, label in neighbors:
                    votes[label] = votes.get(label, 0) + 1
                preds.append(max(votes.items(), key=lambda x: x[1])[0])
        return preds

    def to_arduino_code(self, fn_name="predict_knn"):
        if not self.X_train: return "// Error: Not trained"
        
        n_samples = len(self.X_train)
        n_features = len(self.X_train[0])
        
        # Aplanar datos X para un array 1D simple
        flat_X = []
        for row in self.X_train: flat_X.extend(row)
        
        dtype_y = "int16_t" if self.task == 'classification' else "float"
        
        code = [
            "// --- MiniML KNN (Flash-based Optimization) ---",
            f"// K={self.k}, Samples={n_samples}, Features={n_features}",
            "// WARNING: High Flash usage. O(N) complexity.",
            "#include <avr/pgmspace.h>",
            ""
        ]
        
        # Guardar todo el dataset en Flash
        code.append(f"const float {fn_name}_X[{len(flat_X)}] PROGMEM = {{{', '.join(map(str, flat_X))}}};")
        code.append(f"const {dtype_y} {fn_name}_y[{n_samples}] PROGMEM = {{{', '.join(map(str, self.y_train))}}};")
        
        # Función predicción
        code.append(f"{dtype_y} {fn_name}(float *row) {{")
        # Arrays locales dinámicos pequeños. 
        # Si N es grande, Bubble Sort en RAM es costoso. 
        # Haremos una búsqueda lineal manteniendo solo los K mejores para no gastar RAM en arrays 'distances'.
        # Estrategia "Priority Queue" in-place simplificada en RAM para ahorrar memoria.
        
        code.append(f"  float top_d[{self.k}];")
        code.append(f"  {dtype_y} top_y[{self.k}];")
        code.append(f"  // Init con infinito")
        code.append(f"  for(int i=0; i<{self.k}; i++) top_d[i] = 3.4028235E38; // Max float")
        
        code.append(f"  for(int i=0; i<{n_samples}; i++) {{")
        code.append(f"    float d = 0.0;")
        code.append(f"    // Calcular distancia leyendo Flash")
        code.append(f"    for(int j=0; j<{n_features}; j++) {{")
        code.append(f"      float tr_val = pgm_read_float(&{fn_name}_X[i*{n_features} + j]);")
        code.append(f"      float diff = row[j] - tr_val;")
        code.append(f"      d += diff * diff;")
        code.append(f"    }}")
        code.append(f"    d = sqrt(d);")
        
        # Inserción ordenada en los top K (mantiene los k menores)
        code.append(f"    // Inserción en lista ordenada de tamaño K")
        code.append(f"    if (d < top_d[{self.k-1}]) {{")
        code.append(f"       int pos = {self.k-1};")
        code.append(f"       while(pos > 0 && d < top_d[pos-1]) {{")
        code.append(f"         top_d[pos] = top_d[pos-1];")
        code.append(f"         top_y[pos] = top_y[pos-1];")
        code.append(f"         pos--;")
        code.append(f"       }}")
        code.append(f"       top_d[pos] = d;")
        # Leer etiqueta de Flash solo si entra en top K
        if self.task == 'classification':
             code.append(f"       top_y[pos] = (int16_t)pgm_read_word(&{fn_name}_y[i]);")
        else:
             code.append(f"       top_y[pos] = pgm_read_float(&{fn_name}_y[i]);")
        code.append(f"    }}")
        code.append(f"  }}")
        
        if self.task == 'regression':
            code.append("  // Average")
            code.append(f"  float sum = 0.0;")
            code.append(f"  for(int i=0; i<{self.k}; i++) sum += top_y[i];")
            code.append(f"  return sum / {self.k}.0;")
        else:
            code.append("  // Majority Vote")
            code.append(f"  {dtype_y} best_label = top_y[0];")
            code.append(f"  int max_count = 0;")
            code.append(f"  for(int i=0; i<{self.k}; i++) {{")
            code.append(f"    int count = 0;")
            code.append(f"    for(int j=0; j<{self.k}; j++) {{")
            code.append(f"      if(top_y[j] == top_y[i]) count++;")
            code.append(f"    }}")
            code.append(f"    if(count > max_count) {{ max_count = count; best_label = top_y[i]; }}")
            code.append(f"  }}")
            code.append(f"  return best_label;")

        code.append("}")
        return "\n".join(code)

# ---------------------------
# Evaluation metrics
# ---------------------------
def accuracy_score(y_true: List[Any], y_pred: List[Any]) -> float:
    if not y_true:
        return 0.0
    return sum(1 for a, b in zip(y_true, y_pred) if a == b) / len(y_true)

def mse(y_true: List[float], y_pred: List[float]) -> float:
    n = len(y_true)
    if n == 0:
        return 0.0
    return sum((a - b) ** 2 for a, b in zip(y_true, y_pred)) / n

def mae(y_true: List[float], y_pred: List[float]) -> float:
    n = len(y_true)
    if n == 0:
        return 0.0
    return sum(abs(a - b) for a, b in zip(y_true, y_pred)) / n

def r2_score(y_true: List[float], y_pred: List[float]) -> float:
    mean_y = sum(y_true) / len(y_true) if y_true else 0.0
    ss_tot = sum((yi - mean_y) ** 2 for yi in y_true)
    ss_res = sum((yi - pi) ** 2 for yi, pi in zip(y_true, y_pred))
    return 1.0 - (ss_res / ss_tot) if ss_tot != 0 else 0.0

# ---------------------------
# Metadata attach helper
# ---------------------------
def attach_metadata(model_obj, metadata: Dict[str, Any]):
    try:
        if not hasattr(model_obj, "metadata"):
            model_obj.metadata = {}
        model_obj.metadata.update(metadata)
    except Exception:
        pass

# ---------------------------
# Self-test (Comprehensive)
# ---------------------------
if __name__ == "__main__":
    print("=== MiniML Runtime Self-Test ===")

    try:
        # 1. Decision Tree Classification
        print("\n[TEST] Decision Tree Classifier...")
        data_cls = [[2.7, 2.5, 0], [1.3, 3.5, 0], [3.5, 1.4, 1], [3.9, 4.0, 1]]
        dt = DecisionTreeClassifier(max_depth=3)
        dt.fit(data_cls)
        pred_dt = dt.predict([[2.5, 2.3], [3.7, 3.9]])
        print(f"  > Predicciones: {pred_dt}")
        print(f"  > C-Code Preview: {dt.to_arduino_code()[:50]}...")

        # 2. Random Forest Regression
        print("\n[TEST] Random Forest Regressor...")
        data_reg = [[1.0, 1.0, 2.0], [2.0, 2.0, 4.0], [3.0, 3.0, 6.0]]
        rf = RandomForestRegressor(n_trees=3, max_depth=3)
        rf.fit(data_reg)
        pred_rf = rf.predict([[4.0, 4.0]])
        print(f"  > Predicción (esperado ~8.0): {pred_rf}")

        # 3. Mini Linear Model
        print("\n[TEST] Mini Linear Model...")
        # y = 2x + 1
        data_lin = [[1.0, 3.0], [2.0, 5.0], [3.0, 7.0], [4.0, 9.0]]
        lm = MiniLinearModel(learning_rate=0.01, epochs=2000)
        lm.fit(data_lin)
        pred_lm = lm.predict([[5.0]])
        print(f"  > Predicción para 5.0 (esperado ~11.0): {pred_lm}")
        print(f"  > Pesos: {lm.weights}")

        # 4. Mini SVM
        print("\n[TEST] Mini SVM...")
        # Clases separables linealmente
        data_svm = [[1.0, 1.0, 1], [2.0, 2.0, 1], [0.0, 0.0, -1], [-1.0, -1.0, -1]]
        svm = MiniSVM(learning_rate=0.01, n_iters=1000)
        svm.fit(data_svm)
        pred_svm = svm.predict([[1.5, 1.5], [-0.5, -0.5]])
        print(f"  > Predicciones (esperado [1, -1]): {pred_svm}")

        # 5. Mini Neural Network (XOR problem)
        print("\n[TEST] Mini Neural Network (XOR)...")
        xor_data = [[0,0,0], [0,1,1], [1,0,1], [1,1,0]]
        X_xor = [r[:-1] for r in xor_data]
        y_xor = [r[-1] for r in xor_data]
        nn = MiniNeuralNetwork(n_inputs=2, n_hidden=4, n_outputs=1, epochs=2000, learning_rate=0.1, seed=42)
        nn.fit(X_xor, y_xor)
        pred_nn = nn.predict([[0,1], [1,1]])
        print(f"  > Predicciones (esperado ~1, ~0): {[round(p[0], 2) for p in pred_nn]}")
        
        # Test Quantization
        nn.quantize()
        print(f"  > Quantized C-Code Preview: {nn.to_arduino_code()[:60]}...")

        # 6. Mini Scaler
        print("\n[TEST] Mini Scaler (Standard)...")
        raw_data = [[10.0, 200.0], [20.0, 400.0], [30.0, 600.0]]
        scaler = MiniScaler(method='standard')
        scaler.fit(raw_data)
        transformed = scaler.transform([20.0, 400.0])
        print(f"  > Transformado (debe estar cerca de 0.0): {transformed}")

        # Validation Logic (Check Dims)
        print("\n[TEST] Validación de Dimensiones (check_dims)...")
        try:
            # Entrenado con 2 features, intentamos predecir con 3
            lm.predict([[1.0, 2.0, 3.0]])
            print("  > FALÓ: No detectó error de dimensiones.")
        except ValueError as e:
            print(f"  > ÉXITO: Detectó error correctamente: {e}")

        print("\n=== Todos los tests de Runtime pasaron correctamente. ===")

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\n[ERROR] El test falló: {e}")