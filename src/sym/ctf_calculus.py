# import networkx as nx
# from typing import Union
# from loguru import logger

# from .dsl import Variable, CounterfactualTerm, Event, Query
# from src.graph import get_mutilated_parents_graph, construct_amwn


# def apply_exclusion(event: Event, G: nx.DiGraph) -> Event:
#     """
#     Rule 3 (Exclusion Rule): Adding/removing interventions.
#     P(y_{xz}, w_*) = P(y_z, w_*) if X intersect An(Y) = empty in G_{bar{Z}}

#     This function iteratively strips out non-ancestral interventions from every
#     counterfactual term in the given event based on the topological constraints.
#     """
#     new_assignments = {}

#     for term, value in event.assignments.items():
#         Y_name = term.variable.name
#         X_dict = term.intervention
#         X_names = {v.name for v in X_dict.keys()}

#         # G_{bar{X}}
#         G_bar = get_mutilated_parents_graph(G, X_names)

#         # An(Y) in G_{bar{X}}
#         ancestors_Y = nx.ancestors(G_bar, Y_name) | {Y_name}

#         # Keep only interventions that are in the ancestral set
#         z_dict = {v: val for v, val in X_dict.items() if v.name in ancestors_Y}

#         reduced_term = CounterfactualTerm(term.variable, z_dict)
#         new_assignments[reduced_term] = value

#         if reduced_term != term:
#             logger.debug(f"Rule 3 (Exclusion): Simplified {term} -> {reduced_term}")

#     return Event(new_assignments)


# def apply_consistency(event: Event) -> Event:
#     """
#     Rule 1 (Consistency Rule): Observation/Intervention exchange.
#     P(y_{T_* x}, x_{T_*}, w_*) = P(y_{T_*}, x_{T_*}, w_*)

#     If we observe X_{T_*} = x, we can drop the intervention 'x' from any
#     other term Y_{T_*, X=x} in the same event.
#     """
#     new_assignments = dict(event.assignments)
#     changed = True

#     while changed:
#         changed = False

#         # Look for an observation X_{T_*} = x
#         for term_x, val_x in list(new_assignments.items()):
#             X_var = term_x.variable
#             T_int = term_x.intervention

#             # Look for another variable Y_{T_*, X=x} = y
#             for term_y, val_y in list(new_assignments.items()):
#                 if term_y == term_x:
#                     continue

#                 # Check if X is intervened on in Y, and its value matches the observed val_x
#                 if X_var in term_y.intervention and term_y.intervention[X_var] == val_x:
#                     # Check if the rest of the intervention strictly matches T_*
#                     rest_int = {
#                         k: v for k, v in term_y.intervention.items() if k != X_var
#                     }

#                     if rest_int == T_int:
#                         # Simplify Y_{T_*, X=x} -> Y_{T_*}
#                         del new_assignments[term_y]
#                         reduced_term = CounterfactualTerm(term_y.variable, T_int)
#                         new_assignments[reduced_term] = val_y
#                         changed = True

#                         logger.debug(
#                             f"Rule 1 (Consistency): Simplified {term_y} -> {reduced_term} given {term_x}={val_x}"
#                         )
#                         break  # Break to avoid modifying dict during iteration
#             if changed:
#                 break

#     return Event(new_assignments)


# def apply_independence(
#     query: Query, G: nx.DiGraph, bidirected_edges: list[tuple[str, str]] = None
# ) -> Query:
#     """
#     Rule 2 (Independence Rule): Adding/removing counterfactual observations.
#     P(y_r | x_t, w_*) = P(y_r | w_*) if (Y_r _|_ X_t | W_*) in G_A.

#     Uses Counterfactual d-separation via the AMWN to drop irrelevant evidence.
#     """
#     current_evidence = dict(query.evidence.assignments)
#     target_terms = list(query.target.assignments.keys())

#     changed = True
#     while changed:
#         changed = False

#         # Evaluate each piece of evidence to see if it can be dropped
#         for x_term in list(current_evidence.keys()):
#             rest_evidence_terms = [k for k in current_evidence.keys() if k != x_term]

#             # Y_* = target U {x_t} U {w_*}
#             Y_star = target_terms + [x_term] + rest_evidence_terms

#             # Construct the AMWN
#             G_A = construct_amwn(G, Y_star, bidirected_edges)

#             # Check counterfactual d-separation
#             is_separated = True
#             for y_term in target_terms:
#                 if not nx.is_d_separator(
#                     G_A, {x_term}, {y_term}, set(rest_evidence_terms)
#                 ):
#                     is_separated = False
#                     break

#             if is_separated:
#                 del current_evidence[x_term]
#                 changed = True
#                 logger.debug(
#                     f"Rule 2 (Independence): Dropped {x_term} from evidence. d-separated from {target_terms} given {rest_evidence_terms}."
#                 )
#                 break  # Restart loop since evidence changed

#     return Query(query.target, Event(current_evidence))


# def simplify_query(
#     query: Query, G: nx.DiGraph, bidirected_edges: list[tuple[str, str]] = None
# ) -> Query:
#     """
#     Applies the ctf-calculus rules iteratively to maximally simplify a query.
#     """
#     changed = True
#     current_query = query

#     while changed:
#         prev_target = current_query.target
#         prev_evid = current_query.evidence

#         # 1. Apply Exclusion to target and evidence events
#         new_target = apply_exclusion(current_query.target, G)
#         new_evid = apply_exclusion(current_query.evidence, G)

#         # 2. Apply Consistency to the joint events
#         # Note: Consistency operates on joint conjunctions, so we evaluate target + evidence together
#         joint_event = new_target & new_evid
#         simplified_joint = apply_consistency(joint_event)

#         # Separate back into target and evidence
#         final_target = Event(
#             {
#                 k: v
#                 for k, v in simplified_joint.assignments.items()
#                 if k in new_target.assignments
#             }
#         )
#         final_evid = Event(
#             {
#                 k: v
#                 for k, v in simplified_joint.assignments.items()
#                 if k in new_evid.assignments
#             }
#         )

#         current_query = Query(final_target, final_evid)

#         # 3. Apply Independence to drop evidence
#         current_query = apply_independence(current_query, G, bidirected_edges)

#         # Check if any changes occurred
#         changed = (current_query.target != prev_target) or (
#             current_query.evidence != prev_evid
#         )

#     return current_query
