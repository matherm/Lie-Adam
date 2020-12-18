import torch.nn as nn
import torch
import numpy as np

def find_topographic_grid(n_components, n_spaces):
    """
    Finds the best embedding grid dimensions (grid_w, grid_h) for embedding `n_spaces` square-like spaces

    Returns
    ------
        grid dimensions (grid_w, grid_h, w, h) and square-like dimensions for the boxes
    """
    n_items = n_components // n_spaces
    perfect_square = int(np.sqrt(n_components)) + 1
    perfect_box = int(np.sqrt(n_items)) + 1

    space_grid_w = np.arange(perfect_square - int(perfect_square * 0.2), perfect_square + int(perfect_square * 0.2))
    space_grid_h = np.arange(perfect_square - int(perfect_square * 0.2), perfect_square + int(perfect_square * 0.2))
    space_w      = np.arange(perfect_box - int(perfect_box * 0.5), perfect_box + int(perfect_square * 0.5))
    space_h      = np.arange(perfect_box - int(perfect_box * 0.5), perfect_box + int(perfect_square * 0.5))
    #print("Search space space_grid_w", space_grid_w.min(), space_grid_w.max())
    #print("Search space space_grid_h", space_grid_h.min(), space_grid_h.max())
    #print("Search space space_w", space_w.min(), space_w.max())
    #print("Search space space_h", space_h.min(), space_h.max())

    def is_solution_valid(grid_w, grid_h, w=1, h=1):
        # Check if grid is large enough
        if grid_w * grid_h >= n_components:
            # Check if boxes have enough items
            if w*h == n_items:
                # Check if boxes have room in grid
                if grid_w > w and grid_h > h: 
                    # Check if width is filled
                    if grid_w % w == 0:
                        return True
        return False

    valid_solutions = []
    for grid_w in space_grid_w:
        for grid_h in space_grid_h:
            for w in space_w:
                for h in space_h:
                    if is_solution_valid(grid_w, grid_h, w, h):
                        valid_solutions.append((grid_w, grid_h, w, h))

    ###################
    # WE HAVE A SET OF VALID SOLUTIONS
    # LETS RANK THEM
    ####################

    def square_loss_grid(grid_w, grid_h):
        return np.abs(grid_w - grid_h)
        
    def square_loss_box(grid_w, grid_h, w, h):
        return np.abs(w - h)
            
    def space_pollution(grid_w, grid_h, w, h):
        per_w = grid_w // w
        per_h = grid_h // h
        return per_w * per_h * n_items - n_components
        
    J = lambda grid_w, grid_h, w, h : square_loss_grid(grid_w, grid_h) + square_loss_box(grid_w, grid_h, w, h) + space_pollution(grid_w, grid_h, w, h)
    
    best_solution, best_loss = (0,0,0,0,0), 1000
    for grid_w, grid_h, w, h in valid_solutions:                    
        loss = J(grid_w, grid_h, w, h)
        if loss < best_loss:
            best_loss = loss
            best_solution = (grid_w, grid_h, w, h)
    
    #print("Best_loss", best_loss, "with solution", best_solution, "space_pollution:",  space_pollution(grid_w, grid_h, w, h))
    return best_solution[:4]


class ISA(nn.Module):

    def __init__(self, n_components, n_subspaces, layout="random"):
        super().__init__()
        
        self.n_components = n_components
        self.n_subspaces = n_subspaces
        
        self.space_size = self.n_components // self.n_subspaces
        self.weight = nn.Parameter(torch.zeros(n_subspaces, n_components), requires_grad=False)
        self.init_nodes_(self.weight, layout=layout)
        
    def init_nodes_(self, weight, layout="topogrid"):
        if layout == "topogrid":
            self.fill_topographic_grid_(weight)
        elif layout == "grid":
            self.fill_grid_(weight)
        elif layout == "random":
            self.fill_random_(weight)
        elif layout == "overlap_grid":
            return self.fill_overlapping_grid_(weight)
        else:
            self.fill_adjacent_(weight)
        assert weight.sum() == self.n_components

    def fill_topographic_grid_(self, weight):
        grid_w, grid_h, w, h = find_topographic_grid(self.n_components, self.n_subspaces)
        grid = np.arange((grid_w*grid_h)).reshape(grid_w, grid_h)
        per_w, per_h = grid_w // w, grid_h // h
        for i in range(self.n_subspaces):
            weight.data[i,:] *= 0  
            idx_h, idx_w =  i // per_w, i % per_w
            selected_idx = grid[idx_h*h:(idx_h+1)* h , idx_w*w:(idx_w+1)* w]
            selected_idx = selected_idx[selected_idx < self.n_components]
            weight.data[i, selected_idx] += 1.
    
    def fill_grid_(self, weight):
        space_size = self.space_size
        used_idx = np.asarray(self.n_components * [0])
        h = int(np.sqrt(self.n_components))
        pts = grid(h)
        neighborhoods = distance_scipy_spatial(pts, k=h*h, metric='cityblock')[1] # (pts, neighbors)
        neighborhoods = np.hstack([np.arange(len(pts)).reshape(-1, 1), neighborhoods]) # add selfloop
        for i in range(self.n_subspaces):
            weight.data[i,:] *= 0  
            available_idx = np.arange(self.n_components)[used_idx == 0]           # available idxs
            pos = min([available_idx[0],len(neighborhoods)-1])
            allowed_nbrs = [ n for n in neighborhoods[pos] if n in available_idx] # allowed neighbors
            selected_idx = allowed_nbrs[:self.space_size]
            weight.data[i,selected_idx] += 1
            used_idx[selected_idx] = 1
            if len(selected_idx) < self.space_size:
                available_idx = np.arange(self.n_components)[used_idx == 0]          
                selected_idx = available_idx[:self.space_size-len(selected_idx)]
                weight.data[i,selected_idx] += 1
                used_idx[selected_idx] = 1

    def fill_overlapping_grid_(self, weight, n=1, loops=10):
        w = int(np.sqrt(self.n_components))
        h = self.n_components // w
        if self.n_components != w*h:
            w = int(np.sqrt(self.n_components//3))
            h = self.n_components // w
        assert self.n_components == w*h
        grid = np.arange(w*h).reshape(h, w)
        grid = np.tile(grid, (3,3)) # replicate the grid for cyclic boundaries (negative indexing)
        if self.n_components == self.n_subspaces:
            idx_xy = np.hstack([ np.repeat(np.arange(h), w).reshape(h*w, 1) + h, np.tile(np.arange(w), h).reshape(h*w,1) + w])
        else:
            idx_xy = np.hstack([ np.repeat(np.arange(h), w).reshape(h*w, 1) + h, np.tile(np.arange(w), h).reshape(h*w,1) + w])
            permut = np.random.permutation(np.arange(len(idx_xy)))
            idx_xy = idx_xy[permut[:self.n_subspaces]]
            idx_xy = idx_xy[np.argsort(idx_xy[:,1], axis=-1, kind="mergesort")] #stable sort by column
            idx_xy = idx_xy[np.argsort(idx_xy[:,0], axis=-1, kind="mergesort")] #stable sort by column
        for i in range(self.n_subspaces): 
            weight.data[i,:] *= 0  
            x, y  = idx_xy[i]
            selected_idx = grid[x-n//2:x+1+n//2, y-n//2:y+1+n//2]
            weight.data[i, selected_idx] += 1
        # repeat in case we missed some inputs
        if weight.sum(0).min() == 0:
            if loops > 0:
                self.fill_overlapping_grid_(weight, n, loops-1)
            else:
                self.fill_overlapping_grid_(weight, n+1, 10)
                print(f"fill_overlapping_grid_() max loops reached. Continue with increased n={n+1} neighboring subspaces.")

    def fill_random_(self, weight):
        used_idx = np.asarray(self.n_components * [0])
        for i in range(self.n_subspaces):
            available_idx = np.arange(self.n_components)[used_idx == 0]
            available_idx = np.random.permutation(available_idx)
            selected_idx  = available_idx[:self.space_size]
            weight.data[i,:] *= 0
            weight.data[i,selected_idx] += 1
            used_idx[selected_idx] = 1
    
    def fill_adjacent_(self, weight):
        used_idx = np.asarray(self.n_components * [0])
        for i in range(self.n_subspaces):
            available_idx = np.arange(self.n_components)[used_idx == 0]
            selected_idx  = available_idx[:self.space_size]
            weight.data[i,:] *= 0
            weight.data[i,selected_idx] += 1
            used_idx[selected_idx] = 1

    def forward(self, X):
        X = torch.pow(X, 2) 
        X = (self.weight @ X.T).T
        X = torch.sqrt(X) 
        return X