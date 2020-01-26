import brain


class Agent:
    def __init__(self , num_states , num_actions):
        self.brain = brain.Brain ( num_states , num_actions )

    def upper_q_function(self):
        self.brain.replay ()

    def get_action(self , state: object , episode: object) -> object:
        action = self.brain.decide_action ( state , episode )
        return action

    def memorize(self , state , action , state_next , reward):
        self.brain.memory.push ( state , action , state_next , reward )
