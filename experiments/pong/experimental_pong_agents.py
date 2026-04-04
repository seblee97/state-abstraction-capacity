#!/usr/bin/env python3
"""
Experimental Pong Agents and Analyzers
=======================================

This module contains experimental classes that are not currently used in the main
analysis pipeline but may be useful for future research:

- HardcodedPongAgent: A physics-based Pong AI that plays optimally
- ActionOptimalSymmetryAnalyzer: Symmetry analyzer based on optimal action equivalence

These classes were developed to explore action-based symmetry detection (whether
symmetric states lead to the same optimal action) as an alternative to state-feature
based symmetry detection.
"""

from typing import Any, Dict, Optional
import gymnasium as gym


class HardcodedPongAgent:
    """
    A hardcoded Pong AI agent that uses physics simulation to play optimally.

    This agent predicts ball trajectory including wall bounces and aims to hit
    the ball with the paddle edge to maximize deflection angle.
    """

    def __init__(self):
        self.BALL_Y_MIN = 44
        self.BALL_Y_MAX = 207
        self.PLAYER_X = 189
        self.PADDLE_HEIGHT = 16

        # Action Map
        self.ACTION_UP = 2
        self.ACTION_DOWN = 3
        self.ACTION_NOOP = 0

    def _predict_impact_y(self, state):
        """Simulates ball trajectory to find intersection with player paddle X plane."""
        ball_x = state["ball_x"]
        ball_y = state["ball_y"]
        ball_dx = state["ball_dx"]
        ball_dy = state["ball_dy"]

        # If ball is lost, not moving, or moving away, go to center
        if ball_x is None or ball_y is None or ball_dx <= 0:
            return 128

        dist_x = self.PLAYER_X - ball_x
        if dist_x < 0: return ball_y # Passed us already

        steps = dist_x / ball_dx
        pred_y = ball_y + (ball_dy * steps)

        # Handle wall bounces
        play_height = self.BALL_Y_MAX - self.BALL_Y_MIN
        rel_y = pred_y - self.BALL_Y_MIN
        num_bounces = int(rel_y // play_height)
        remainder = rel_y % play_height

        if num_bounces % 2 == 0:
            final_y = self.BALL_Y_MIN + remainder
        else:
            final_y = self.BALL_Y_MAX - remainder

        return final_y

    def get_action_exploiter(self, state):
        """
        TIER 3: The Physics Exploiter (Safer Version).
        Aims to hit the ball with the corner to maximize deflection angle,
        but with a larger safety margin to prevent 'whiffing'.
        """
        if state.get("ball_dy") is None or state.get("ball_x") is None:
            return self.ACTION_NOOP

        impact_y = self._predict_impact_y(state)
        player_y = state["player_y"]
        ball_dy = state["ball_dy"]

        # Offset: Distance from the edge.
        # 0 = Extreme Edge (Risky), 8 = Center (Safe).
        # Changed from 4 to 6 to increase hit rate.
        offset = 6

        if ball_dy < 0: # Ball moving UP
            # Hit with upper half to deflect up
            target_pos = impact_y - offset
        elif ball_dy > 0: # Ball moving DOWN
            # Hit with lower half to deflect down
            target_pos = impact_y - self.PADDLE_HEIGHT + offset
        else:
            target_pos = impact_y - (self.PADDLE_HEIGHT / 2)

        # Recover to center if ball is moving away
        if state.get("ball_dx", 0) < 0:
            target_pos = 128

        # Hysteresis band (2px)
        if player_y < target_pos - 2:
            return self.ACTION_DOWN
        elif player_y > target_pos + 2:
            return self.ACTION_UP
        else:
            return self.ACTION_NOOP


class ActionOptimalSymmetryAnalyzer:
    """
    Symmetry analyzer that determines symmetry based on optimal action equivalence.

    Instead of checking if states have similar features, this analyzer checks if
    a hardcoded optimal agent would take the same action in both states.

    Note: Requires PongSymmetryAnalyzer as a parent class if you want to use
    the full functionality. This is a standalone version for reference.
    """

    def __init__(self, render_mode: Optional[str] = None):
        # Override init to use the DETERMINISTIC environment
        # PongNoFrameskip-v4 has NO sticky actions and stable physics.
        self.env = gym.make(
            "PongNoFrameskip-v4",
            obs_type="ram",
            render_mode=render_mode
        )

        # Copy constants from parent (re-declaring purely for safety)
        self.PONG_RAM_INDEX = {"ball_x": 49, "ball_y": 54, "enemy_y": 50, "player_y": 51}
        self.BALL_Y_MID = (207 - 44) / 2

        self.agent = HardcodedPongAgent()
        self.sampled_states = []

    def action_equivalence(self, state_1: Dict[str, Any], state_2: Dict[str, Any]) -> bool:
        """Returns True if the Agent selects the exact same action."""
        action_1 = self.agent.get_action_exploiter(state_1)
        action_2 = self.agent.get_action_exploiter(state_2)
        return action_1 == action_2

    def generate_similarity_matrix(self, states=None, symmetry_function=None):
        """
        Generate similarity matrix using action equivalence.

        Note: This method assumes a parent class with generate_similarity_matrix.
        If using standalone, you'll need to implement the full logic.
        """
        if symmetry_function is None:
            symmetry_function = self.action_equivalence
        # Would call: return super().generate_similarity_matrix(states, symmetry_function)
        raise NotImplementedError("Requires PongSymmetryAnalyzer parent class")
