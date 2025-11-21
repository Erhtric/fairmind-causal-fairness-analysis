

def total_effect(x0, x1, y, P_y_given_x_z, P_z):
    effect = 0.0
    for z, pz in P_z.items():
        first_el = P_y_given_x_z(y, x1, z)
        second_el = P_y_given_x_z(y, x0, z)
        effect += (first_el - second_el) * pz
    return effect


def spurious_effect(x, y, P_y_given_x_z,P_z, P_z_given_x):
    effect = 0.0
    for z, pz in P_z.items():
        first_f = P_y_given_x_z(y, x, z)
        second_f = pz- P_z_given_x(z, x)
        effect += (first_f * second_f)
    return effect

def dir_effect(x0,x1,y,W_values, P_y_given_x_z_w,P_z, P_w_given_x_z):
    effect = 0.0
    for z, pz in P_z.items():
      inner_sum=0.0
      for w in W_values:
        first_f= P_y_given_x_z_w(y, x1, z,w)-P_y_given_x_z_w(y, x0, z,w)
        second_f = P_w_given_x_z(x0, z,w)
        inner_sum+=(first_f*second_f)
      effect += inner_sum * pz
    return effect

def ind_effect(x1, x0, y,W_values, P_y_given_x_z_w, P_w_given_x_z,P_z):
    effect = 0.0
    for z, pz in P_z.items():
      inner_sum=0.0
      for w in W_values:
        first_f = P_y_given_x_z_w(y, x1, z, w)
        second_f= P_w_given_x_z( x0,z,w) - P_w_given_x_z(x1, z,w)
        inner_sum += first_f * second_f
      effect += inner_sum * pz
    return effect

def ind_f_effect(x0, x1, y,W_values, P_y_given_x_z_w, P_w_given_x_z,P_z):
    effect = 0.0
    for z, pz in P_z.items():
      inner_sum=0.0
      for w in W_values:
        first_f = P_y_given_x_z_w(y, x0, z, w)
        second_f= P_w_given_x_z( x1, z,w) - P_w_given_x_z(x0, z,w)
        inner_sum += first_f * second_f
      effect += inner_sum * pz
    return effect