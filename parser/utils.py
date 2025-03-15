import torch
import pyscipopt as scp
from parser.utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_FEATURES = 6                # число базовых признаков переменной
EMBEDDING_FEATURES = 15          # дополнительное число признаков (не используется напрямую)
INITIAL_MIN = 1e+20              # начальное минимальное значение для коэффициентов
CLIP_EPS = 1e-5                # нижняя граница для clamping при нормализации
CLIP_MAX_OBJ = 20000             # верхняя граница для коэффициента цели
CLIP_MIN_OBJ = 0                 # нижняя граница для коэффициента цели

def load_model(instance_path: str) -> scp.Model:
    model = scp.Model()
    model.hideOutput(True)
    model.readProblem(instance_path)
    return model

def extract_variable_features(model: scp.Model, base_features=BASE_FEATURES, init_min=INITIAL_MIN):
    variables = model.getVars()
    variables.sort(key=lambda var: var.name)
    var_features = []
    binary_var_indices = []
    for idx, var in enumerate(variables):
        # [0]: коэффициент цели,
        # [1]: сумма нормированных коэффициентов ограничений,
        # [2]: степень (число появлений в ограничениях),
        # [3]: максимальный коэффициент,
        # [4]: минимальный коэффициент,
        # [5]: флаг бинарности
        features = [0] * base_features
        features[3] = 0
        features[4] = init_min
        if var.vtype() == 'BINARY':
            features[base_features - 1] = 1
            binary_var_indices.append(idx)
        var_features.append(features)
    var_name_to_index = {var.name: idx for idx, var in enumerate(variables)}
    return variables, var_features, binary_var_indices, var_name_to_index

def process_objective(model: scp.Model, var_features, var_name_to_index):
    objective = model.getObjective()
    # indices: [строки, столбцы] для разреженной матрицы
    indices = [[], []]
    values = []
    # Признаки целевой функции (для дополнительного узла)
    obj_feature = [0, 0, 0, 0]
    for term in objective:
        var_name = term.vartuple[0].name
        coeff = objective[term]
        var_idx = var_name_to_index[var_name]
        var_features[var_idx][0] = coeff 
        if coeff != 0:
            indices[0].append(0) 
            indices[1].append(var_idx)
            values.append(1)
        obj_feature[0] += coeff
        obj_feature[1] += 1
    if obj_feature[1] > 0:
        obj_feature[0] /= obj_feature[1]
    else:
        obj_feature[0] = 0
    return indices, values, obj_feature

def process_constraints(model: scp.Model, var_features, var_name_to_index, indices, values, obj_feature):
    all_cons = model.getConss()
    # Отбираем только ограничения с линейными коэффициентами
    filtered_cons = [con for con in all_cons if len(model.getValsLinear(con)) > 0]
    n_cons = len(filtered_cons)
    # Сортируем ограничения по числу коэффициентов и имени (для стабильности)
    cons_info = [[con, len(model.getValsLinear(con))] for con in filtered_cons]
    cons_info.sort(key=lambda x: (x[1], str(x[0])))
    constraints = [x[0] for x in cons_info]

    cons_features = []
    for con_idx, con in enumerate(constraints):
        coeffs = model.getValsLinear(con)
        rhs = model.getRhs(con)
        lhs = model.getLhs(con)
        # Определяем тип ограничения:
        # sense = 2: равенство, sense = 1: неравенство (при rhs >= BIG_NUMBER), sense = 0: прочее
        if rhs == lhs:
            sense = 2
        elif rhs >= INITIAL_MIN:
            sense = 1
            rhs = lhs
        else:
            sense = 0

        coeff_sum = 0
        for var_name, coeff in coeffs.items():
            var_idx = var_name_to_index[var_name]
            if coeff != 0:
                indices[0].append(con_idx)
                indices[1].append(var_idx)
                values.append(1)
            # Обновляем признаки переменной:
            # [2]: степень (кол-во появлений), [1]: сумма нормированных коэффициентов,
            # [3]: максимальный коэффициент, [4]: минимальный коэффициент
            var_features[var_idx][2] += 1
            var_features[var_idx][1] += coeff / max(n_cons, 1)
            var_features[var_idx][3] = max(var_features[var_idx][3], coeff)
            var_features[var_idx][4] = min(var_features[var_idx][4], coeff)
            coeff_sum += coeff
        num_coeffs = max(len(coeffs), 1)
        cons_features.append([coeff_sum / num_coeffs, len(coeffs), rhs, sense])
    # Добавляем признаки целевой функции как дополнительное ограничение
    cons_features.append(obj_feature)
    return cons_features, n_cons

# Функция нормализации тензоров
def normalize_tensor(tensor: torch.Tensor, lower_bound=CLIP_EPS, upper_bound=1.0) -> torch.Tensor:
    max_vals, _ = torch.max(tensor, dim=0)
    min_vals, _ = torch.min(tensor, dim=0)
    diff = max_vals - min_vals
    diff[diff == 0] = 1
    norm_tensor = (tensor - min_vals) / diff
    norm_tensor = torch.clamp(norm_tensor, lower_bound, upper_bound)
    return norm_tensor

def get_problem_data(instance_path: str):
    model = load_model(instance_path)
    n_vars = model.getNVars()

    variables, var_features, binary_var_indices, var_name_to_index = extract_variable_features(model, BASE_FEATURES, INITIAL_MIN)

    indices, values, obj_feature = process_objective(model, var_features, var_name_to_index)
    cons_features, n_cons = process_constraints(model, var_features, var_name_to_index, indices, values, obj_feature)

    var_features_tensor = torch.tensor(var_features, dtype=torch.float32, device=device)
    cons_features_tensor = torch.tensor(cons_features, dtype=torch.float32, device=device)
    binary_var_tensor = torch.tensor(binary_var_indices, dtype=torch.int32, device=device)

    indices_tensor = torch.tensor(indices, dtype=torch.int64)
    values_tensor = torch.tensor(values, dtype=torch.float32)
    A = torch.sparse_coo_tensor(indices_tensor, values_tensor, size=(n_cons + 1, n_vars)).to(device)

    var_features_tensor[:, 0] = torch.clamp(var_features_tensor[:, 0], CLIP_MIN_OBJ, CLIP_MAX_OBJ)
    var_features_tensor = normalize_tensor(var_features_tensor, CLIP_EPS, 1.0)

    cons_features_tensor = normalize_tensor(cons_features_tensor, CLIP_EPS, 1.0)

    return A, var_name_to_index, var_features_tensor, cons_features_tensor, binary_var_tensor

if __name__ == '__main__':
    instance_path = 'lp_problems/a.lp'
    A, var_map, var_features, cons_features, binary_vars = get_problem_data(instance_path)
    print(A)
    print(var_map)
    print(var_features)
    print(cons_features)
    print(binary_vars)