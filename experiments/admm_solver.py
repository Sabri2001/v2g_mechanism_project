import logging
from copy import deepcopy, copy
import numpy as np

class ADMM:
    """
    A generic ADMM solver that delegates local subproblems and global updates
    to user-provided callback functions.
    """

    def __init__(
        self,
        num_agents: int,
        nb_time_steps: int,
        nu: float = 0.1,
        nu_multiplier: float = 1.2,
        max_iter: int = 100,
        tol: float = 1e-3,
        local_subproblem_fn=None,
        global_step_fn=None
        ):
        """
        Args:
            num_agents (int): Number of agents (EVs).
            nb_time_steps (int): Number of time steps.
            nu (float): ADMM penalty parameter.
            nu_multiplier (float): ADMM penalty parameter multiplier, in adaptive step size rule.
            max_iter (int): Maximum number of ADMM iterations.
            tol (float): Convergence tolerance for dual changes.
            local_subproblem_fn (callable): A callback with signature (agent_idx, iteration_state).
            global_step_fn (callable): A callback with signature (iteration_state).
        """
        self.num_agents = num_agents
        self.nb_time_steps = nb_time_steps
        self.nu = nu
        self.nu_multiplier = nu_multiplier
        self.max_iter = max_iter
        self.tol = tol

        if local_subproblem_fn is None:
            raise ValueError("Must provide local_subproblem_fn to solve local subproblems.")
        self.local_subproblem_fn = local_subproblem_fn

        if global_step_fn is None:
            raise ValueError("Must provide global_step_fn to update global variables.")
        self.global_step_fn = global_step_fn

        # iteration_state is a dictionary that the user is free to structure.
        # Typically it includes primal/dual variables, references to data, etc.
        self.iteration_state = {}
        self.old_iteration_state = {}

    def solve(self):
        """
        Executes the ADMM iteration loop, calling:
          1) local_subproblem_fn for each agent,
          2) global_step_fn once per iteration,
          3) checks for convergence based on dual changes.

        Returns:
            dict: The final iteration_state dictionary after ADMM converges or reaches max_iter.
        """
        self.iter_count = 0

        for iteration in range(self.max_iter):
            logging.debug(f"\n=== ADMM Iteration {iteration} ===")

            old_dual = self._snapshot_of_duals()

            # 1) Solve local subproblem for each agent
            for i in range(self.num_agents):
                self.local_subproblem_fn(i, self.iteration_state, self.old_iteration_state)

            # 2) Global step
            self.global_step_fn(self.iteration_state)
            self.old_iteration_state = copy(self.iteration_state) # shallow copy

            # 3) Convergence check
            new_dual = self._snapshot_of_duals()
            logging.debug(f"new_dual: {new_dual[:5]}")
            dual_diff = self._compute_dual_diff(old_dual, new_dual)
            logging.debug(f"Dual difference: {dual_diff}")

            if dual_diff < self.tol:
                logging.info(f"ADMM converged after {iteration} iterations.")
                self.iter_count = iteration + 1
                break
            else:
                self.nu *= self.nu_multiplier  # Increase penalty parameter for better convergence

        if iteration == self.max_iter - 1:
            logging.warning(f"ADMM reached max_iter={self.max_iter} without converging.")
            self.iter_count = self.max_iter
            
        return self.iteration_state

    def _snapshot_of_duals(self):
        """
        Returns a copy of whichever dual variables you want to monitor.
        By default, we assume 'dual' might be your dual in iteration_state.
        """
        if "dual" in self.iteration_state:
            return self.iteration_state["dual"].copy()
        return None

    def _compute_dual_diff(self, old_dual, new_dual):
        """
        Computes the norm difference in dual variables for convergence checks.
        """
        if old_dual is None or new_dual is None:
            return 999999  # effectively no convergence check
        return np.linalg.norm(new_dual - old_dual)
