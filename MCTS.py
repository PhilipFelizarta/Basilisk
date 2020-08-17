import numpy as np
import env

from tqdm import tqdm

def muMCTS(board, turn, model, simulations=10000, dirichlet=False, alpha=10.0/(63.0), verbose=False):
    if simulations == 0:
        pos = env.convert(board)
        pos = np.reshape(pos, [-1, 8, 8, 17])
        policy, value = model.predict(pos)
        mask = env.legal_bitmask(board)
        policy = policy * mask
        policy = policy/np.sum(policy)
        return policy, value
    
    pos = env.convert(board)
    pos = np.reshape(pos, [-1, 8, 8, 17])
    policy, init_value = model.predict(pos)
    mask = env.legal_bitmask(board)
    policy = policy * mask
    policy = policy/np.sum(policy)
    
    
    init_policy = policy
    new_priors = np.power(policy, 1/1.0)
    policy = new_priors/np.sum(new_priors)
    if dirichlet:
        dirich = np.random.dirichlet([alpha] * 1924)
        policy = (policy * 0.75) + (0.25 * dirich)
    root = init_root(board.copy(), policy, turn, np.squeeze(init_value), mask)
    
    for i in range(simulations):
        if verbose:
            print("Simulation: ", i)
            print(root)
            
        leaf_parent, leaf_board, action = root.leaf()
        
        uci_move = env.MOVES[action]
        new_board = leaf_board.copy()
        if new_board.is_game_over():
            value = 0
            if board.is_checkmate():
                if board.turn:
                    value = -1
                else:
                    value = 1
            leaf_parent.backup(value)
        else:
            new_board.push_uci(uci_move)
            if new_board.is_game_over():
                value = 0
                if board.is_checkmate():
                    if board.turn:
                        value = -1
                    else:
                        value = 1
                policy = np.zeros(1924)
                leaf_parent.expand_backup(action, new_board, policy, value, policy)
            else:
                pos = env.convert(new_board)
                pos = np.reshape(pos, [-1, 8, 8, 17])
                policy, value = model.predict(pos)
                mask = env.legal_bitmask(new_board)
                policy = policy * mask
                policy = policy/np.sum(policy)
                new_priors = np.power(policy, 1/1.0)
                policy = new_priors/np.sum(new_priors)
                leaf_parent.expand_backup(action, new_board, policy, np.squeeze(value), mask)
            if verbose:
                clear_output(wait=True)

    #val = root.child_Q()[np.argmax(root.child_plays)] #Return the q value of our most visited node
    if verbose:
        print(root)
        
    return root.child_plays/np.sum(root.child_plays), root.child_Q(), init_policy

def init_root(hidden_state, policy, turn, fpu, mask): #Handle root case
    root = Node(None, 0, hidden_state, policy, turn, fpu, mask)
    return root

class Node:
    def __init__(self, parent, index, state, policy, turn, fpu, mask): #On init we need to define the parent and the index the node is in the parent child array
        self.parent = parent
        self.index = index
        self.turn = turn #Boolean to switch our pUCT conditions
        self.policy = policy
        self.state = state

        fpu_red = 0.012
        if not turn:
            fpu_red = -0.012

        self.mask = mask
        self.child_plays = np.zeros([1924], dtype=np.int32) #Keep track of how many times our children have played
        self.child_values = np.full([1924], np.clip(fpu - fpu_red, -1, 1), dtype=np.float32) #Keep track of the sum of q values

        self.children = [None]*1924 #A list of children, there will 1924 of them.. no python chess to tell us less children

    def child_Q(self): #return average Q-values
        values = self.child_values
        if self.turn:
            values = (1-self.mask)*-100 + values
        else:
            values = (1-self.mask)*100 + values
            
        return values / (1 + self.child_plays)

    def child_U(self): #return puct bound
        #DEFINE HYPERPARAMETERS HERE
        c1 = 1.2
        c2 = 19652

        #Define sum of plays among the children
        total_plays = np.sum(self.child_plays)
        u = (c1 + np.log((total_plays + c2 + 1)/c2)) * np.sqrt(total_plays + 1) / (1 + self.child_plays)
        return self.policy * u

    def pUCT_child(self): #Returns state action pair (s', a)
        #print("calc puct")
        if self.turn: #CT
            child_index = np.argmax(self.child_Q() + self.child_U())
            return self.children[child_index], child_index
        else: #T
            child_index = np.argmin(self.child_Q() - self.child_U())
            return self.children[child_index], child_index

    def leaf(self, depth_max=10):
        #print("finding leaf")
        current = self
        parent = self
        depth = 0
        while current is not None:
            parent = current
            current, action = current.pUCT_child()
            depth += 1
        return parent, parent.state, action #Action must be converted to one-hot or other formatting in search function

    def expand_backup(self, index, new_state, policy, value, mask): #Create a child at index with state and policy
        #print("expanding")
        child = Node(self, index, new_state, policy, not self.turn, value, mask)
        self.children[index] = child
        self.child_values[index] += value
        self.child_plays[index] += 1
        self.backup(value)

    def backup(self, value):
        #print("backing")
        current = self
        while current.parent != None:
            current.parent.child_values[current.index] += value
            current.parent.child_plays[current.index] += 1
            current = current.parent
    
    def __repr__(self, level=0, norm=0, play=0): #Print tree
        if norm == 0:
            norm = np.sum(self.child_plays)
            play = norm
            
        norm_plays = self.child_plays/norm
        top_norm = np.sum(self.child_plays)
        
        act = env.MOVES[self.index]
        
        ret = "\t"*level+act+"("+str(self.index)+"): "+str(play)+"\n"
        for child in self.children:
            if child is not None:
                if norm_plays[child.index] >= 0.02:
                    ret += child.__repr__(level=level+1, norm=norm, play=self.child_plays[child.index])
        return ret
