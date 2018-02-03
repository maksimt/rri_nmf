def universal_stopping_condition(obj_history, eps_stop=1e-4):
    """ Check if last change in objective is <= eps_stop * first change"""
    if len(obj_history) < 2:
        return False  # dont stop

    d1 = abs(obj_history[0] - obj_history[1])
    de = abs(obj_history[-1] - obj_history[-2])
    return de <= eps_stop * d1


def first_last_stopping_condition(obj_history, eps_stop=1e-4):
    if len(obj_history) < 2:
        return False
    return obj_history[-1] <= obj_history[0] * eps_stop