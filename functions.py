import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from numba import njit, prange
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.patches import Circle

'''
list of functions:
    sample_uniform_disk
    _seed_numba_rng
    _sample_point_from_center
    _grow_numba
    _assign_best_center
    run_growth
    P_star_theory           
    employment_density_grid
    plot_density_snapshot
    plot_density_evolution
    _data_per_point
    plot_arrows_snapshot
    plot_arrows_evolution   
    k_of_P_curve_numpy
    k_of_P_curve_numba
'''

def sample_uniform_disk(n, R):
    '''sample n points uniformly in a disk of radius R in cartesian coordinates

    inverse sampling: X~F^-1(U(0,1))   F is the cdf
    F(r) ~ r^2/R^2  =>  r ~ R*sqrt(U(0,1))'''

    r = R * np.sqrt(np.random.random(n))        
    theta = 2 * np.pi * np.random.random(n)            #casual n angles
    return r * np.cos(theta), r * np.sin(theta)        #result in cartesian coordinates

@njit
def _seed_numba_rng(seed):
    '''
    in numba it is not possible to use np.random.seed(seed) called from
    outside the numba function, so we need to create a function that sets the seed inside numba
    '''
    np.random.seed(seed)


@njit
def _sample_point_from_center(cx_j, cy_j, r0_ker, R):
    '''sample one point from the exponential radial kernel exp(-r/r0_ker)
    centered at (cx_j, cy_j), truncated to the city disk of radius R (rejection sampling).
    r ~ Gamma(shape=2, scale=r0_ker)'''
    while True:                                  # 
       r = np.random.gamma(2.0, r0_ker)          # extract r
       theta = 2 * np.pi * np.random.random()    # extract theta
       x = cx_j + r * np.cos(theta)              # to cartesian coordinates
       y = cy_j + r * np.sin(theta)
       if x*x + y*y <= R*R:                      # accept only if inside the disk
           return x, y                           # if false, it will repeat the loop bcs loop condition always True

@njit
def _assign_best_center(cx, cy, eta, T, xt, yt, l, c, mu, N_c):
    '''computes Z_ij for a single worker located at (xt, yt) against every center j,
    and returns the index of the best one (argmax_j Z_ij)'''
    inv_l = 1/l
    inv_c = 1/c
    best_j,best_z = 0,-1e300
    for j in range(N_c):
        dx,dy = cx[j]-xt, cy[j]-yt
        d = np.sqrt(dx*dx+dy*dy)
        z=eta[j]-(d*inv_l)*(1+(T[j]*inv_c)**mu)
        if z>best_z:
            best_z,best_j=z,j
    return best_j


@njit
def _grow_numba(cx, cy, eta, T, wx, wy, l, c, mu, P_max, N_c, R, r0_ker, sampling: str):
    '''it assigns each worker to the best center, 
    and updates the number of workers assigned to that center (traffic)
 
    sampling = 'uniform' : worker t assigned uniformely, calling sample_uniform_disk
    sampling = 'clark'   : worker t sampled from the current kde of Clark profile;
                         wx,wy are initialized empty 
    
    cx,cy    : coordinates of the centers
    T        : traffic, number of workers assigned to each center (updated in place)
    wx,wy    : coordinates of the workers (sampling='uniform': read as input;
               sampling='clark': filled in place as worker positions are generated)
    R        : city radius 
    r0_ker   : kernel radius of the Clark profile 
    '''
    assignment = np.empty(P_max, dtype=np.int32)   # it will store the index of the center assigned to each worker
 
    for t in range(P_max):                         # a time step is the assignment of a new worker
        if sampling == 'uniform':                  
            xt, yt = wx[t], wy[t]                  # coordinates of the worker t saved as temporary variables for accessing memory only once
        
        elif sampling == 'clark':   
            '''
            since the pdf is a kde, I first sample a center j with probability proportional to its traffic T[j], 
            and then sample a point from the exponential kernel centered at that center
            '''
            u = np.random.random() * t      # random number in (0,t); t is the actual population
                                            # select the center whose cumulative weight contains 'u'
            center = 0
            cum_weight = T[0]          

            while u >= cum_weight:        # in which center the random number falls
                center += 1
                cum_weight += T[center]

            src = center

            xt, yt = _sample_point_from_center(cx[src], cy[src], r0_ker, R)
            wx[t], wy[t] = xt, yt           # worker living coordinates
 
        best_j = _assign_best_center(cx, cy, eta, T, xt, yt, l, c, mu, N_c)  # compute Z_ij for each center j and keep the best
        assignment[t] = best_j
        T[best_j] += 1.0                           # the traffic of that center increases by 1

    return wx, wy, assignment


def run_growth(P_max, N_c, R, l, c, mu,sampling:str='uniform',seed=None,k0_ker=None):
    """
    fixing initial conditions and  then running the growth of the city by assigning workers to centers
    """
    cx = np.empty(N_c)                               # it will store the coordinates of the centers
    cy = np.empty(N_c)
    cx[0], cy[0] = 0.0, 0.0                          # fixing the first center at the origin
    cx[1:], cy[1:] = sample_uniform_disk(N_c - 1, R) # calling function to sample in cartesian coordinates the other centers

    eta = np.random.random(N_c)                      # assigning eta for each center
    T = np.zeros(N_c)                                # traffic initially at zero for each center
    if sampling == 'uniform':
        wx, wy = sample_uniform_disk(P_max, R)           # assigning uniformly the workers
    elif sampling == 'clark':
        T[0] = 1.0                                      # first worker assigned to the first center
        wx, wy = np.empty(P_max), np.empty(P_max)       # initializing empty arrays
        _seed_numba_rng(seed) 
                                            # setting the seed for reproducibility
    wx,wy,assignment = _grow_numba(cx, cy, eta, T, wx, wy, l, c, mu, P_max, N_c,R,k0_ker,sampling)     # assigning to each worker the best center, and updating the traffic of each center

    return cx, cy, eta, wx, wy, assignment


def P_star_theory(l, L, N_c, c, mu):
    '''analytical prediction of the critical population P* at which the system becomes polycentric'''
    return c * (l / (L * N_c)) ** (1.0 / mu)


@njit(parallel=True)
def employment_density_grid(cx, cy, T, R, N_c, n_bins, r0_ker):
    """
    KDE traffic profile (T_R(j) kernel) for the plot
    """
    edges = np.linspace(-R, R, n_bins + 1)            # edges of the bins
    xc = 0.5 * (edges[:-1] + edges[1:])               # centers of the bins
    r0_ker_inv = 1.0 / r0_ker                         # inverse of the kernel radius for efficiency
    density_grid = np.zeros((n_bins, n_bins))         # it will store the density of workers in each bin

    for j in range(N_c):                              # for each center   
        T_j=T[j]                                      # T : traffic array
        if T_j <= 0:
            continue
        cx_j, cy_j = cx[j], cy[j]       
        for i in prange(n_bins):                      # for each bin in x direction, 'ik' indexing
            dx = xc[i] - cx_j                         # distance bin-center in x direction
            dx2 = dx * dx
            for k in range(n_bins):                   # for each bin in y direction
                dy = xc[k] - cy_j                     # grid is square, so we can use xc for y as well
                d = np.sqrt(dx2 + dy * dy)
                density_grid[i, k] += T_j * np.exp(-d * r0_ker_inv) 
    return density_grid, edges




def plot_density_snapshot(ax, cx, cy, assignment, t, N_c, R, n_bins, r0_ker,
                           vmin, vmax, cmap_name, color_scale):
    '''
    for creating a single subplot
    ax              matplotlib.axes.Axes
    cx,cy           coordinates of the centers
    assignment      array of length P_max, with the indices of the centers
    t               time step (number of workers assigned)
    N_c             number of centers
    R               city radius
    n_bins          number of bins for the grid
    r0_ker          kernel radius for the KDE
    vmin, vmax      min and max values for the color scale
    cmap_name       name of the colormap
    color_scale     "linear" or "log"
    '''

    T_t = np.bincount(assignment[:t], minlength=N_c).astype(float)           # array of length N_c, with the number of workers assigned to each center at time t
    density_grid, edges = employment_density_grid(cx, cy, T_t, R, N_c, n_bins, r0_ker)  #calculating grid at time t
    xc = 0.5 * (edges[:-1] + edges[1:])        # centers of the bins, equal in x and y 
    XX, YY = np.meshgrid(xc, xc, indexing="ij")# grid of the centers of the bins, for masking the outside of the city
    outside = (XX*XX + YY*YY) > R*R            # a boolean array of shape (n_bins, n_bins): True if outside
    density_plot = np.ma.masked_where(outside, density_grid) # it delates density_grid where outside is True

    if color_scale == "log":
        density_plot = np.ma.masked_where(density_plot <= 0, density_plot) # prevent log(0) error
        im = ax.pcolormesh(edges, edges, density_plot.T, cmap=cmap_name,
                            norm=LogNorm(vmin=max(vmin, 1e-6), vmax=vmax)) #vmin for preventing log(0) error; norm necessary for log scale
    elif color_scale == "linear":
        im = ax.pcolormesh(edges, edges, density_plot.T, cmap=cmap_name,vmin=vmin, vmax=vmax)

    ax.axis('off')
    k_active = np.count_nonzero(T_t > 0)  # count the number of active centers at time t
    ax.set_title(f"P = {t:,}\nk = {k_active}/{N_c} active centers", fontsize=10) # t:, for thousands separator
    return im


def plot_density_evolution(cx, cy, assignment, N_c, R, n_bins, r0_ker, P_max,
                            n_snapshots, color_scale, cmap_name):
    
    P_init = 1
    snapshots = np.linspace(P_init, P_max, n_snapshots, dtype=int) #how many people at each snapshot        

    T_final = np.bincount(assignment, minlength=N_c).astype(float) #traffic at final step
    density_final, _ = employment_density_grid(cx, cy, T_final, R, N_c, n_bins, r0_ker)
    
    vmin_common = 0.0 if color_scale == "linear" else 1e-6     # common color scale values for each subplot
    vmax_common = density_final.max()

    n_cols = int(np.ceil(np.sqrt(n_snapshots)))                # ceils rounds up
    n_rows = int(np.ceil(n_snapshots / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.3*n_cols, 4.3*n_rows))
    axes_flat = np.atleast_1d(axes).flatten()         # transforms to array and flatten for loop; atleast_1d for preventing error when n_snapshots=1 

    for ax, t in zip(axes_flat, snapshots):                   
        im = plot_density_snapshot(ax, cx, cy, assignment, t, N_c, R, n_bins, r0_ker,
                                    vmin_common, vmax_common, cmap_name, color_scale)
    for ax in axes_flat[len(snapshots):]:    # for extra axes when n_snapshots < n_cols*n_rows
        ax.axis("off")

    cbar = fig.colorbar(im, ax=axes_flat.tolist(), fraction=0.025, pad=0.02) #fraction referred to % width of the colorbar wrt the whole figure, pad to % distance
    cbar.set_label(f"employment density, {color_scale} scale)")
    fig.suptitle(f"Employment density over time - grid {n_bins}x{n_bins}, "
                 f"r0_ker={r0_ker:.2f} km, N_c={N_c}", fontsize=12, y=1.0)
    plt.show()


'''
SIMPLE VERSION OF THE ARROWS PLOTS BUT MUCH SLOWER


def plot_arrows_snapshot(ax, cx, cy, wx, wy, assignment, t, R, N_c, n_workers_per_snapshot, cmap_name):
    """
    Arrow from each sampled worker's residence to the center they chose
    """
    sub_assign = assignment[:t]                # assignment of the first t workers
    n_show = min(t, n_workers_per_snapshot)    # number of workers to show in the plot
    np.random.seed(0)                           
    idx = np.random.choice(t, n_show, replace=False)     # choose n_show from t workers 

    cmap = plt.get_cmap(cmap_name)                       # a function that will take an argument (index)
    n_colors = cmap.N if hasattr(cmap, "N") else 10      # number of colors in the colormap; hasattr: has attribute
    center_colors = [cmap(j % n_colors) for j in range(N_c)] # list of colors for each center, cycling through the colormap if N_c > n_colors

    for i in idx:                    # for each sampled worker
        j = sub_assign[i]            # for each subcenter
        wx_i, wy_i = wx[i], wy[i]    
        cx_j, cy_j = cx[j], cy[j]
        ax.annotate(                                   # first argument is text ("")          
            "", xy=(cx_j, cy_j), xytext=(wx_i, wy_i),  # 2nd and 3rd arguments: to and from coordinates of the arrow
            arrowprops=dict(arrowstyle="->", color=center_colors[j],
                             alpha=arrow_alpha, lw=arrow_width),
        )

    ax.add_patch(plt.Circle((0, 0), R, fill=False, linestyle="--", color="gray", linewidth=0.8))
    ax.set_xlim(-R, R); ax.set_ylim(-R, R)   #ax. : it modifies the axes created before; return not needed
    ax.set_aspect("equal")
    ax.axis('off')
    k_active = np.count_nonzero(np.bincount(sub_assign, minlength=N_c))
    ax.set_title(f"P = {t:,}\nk = {k_active}/{N_c} active centers", fontsize=10)


def plot_arrows_evolution(cx, cy, wx, wy, assignment, N_c, R,
                           n_snapshots=6, n_workers_per_snapshot=600,
                           cmap_name="tab10", P_max=P_max):
  
    P_init = 1
    snapshots = np.linspace(P_init, P_max, n_snapshots, dtype=int)

    n_cols = int(np.ceil(np.sqrt(n_snapshots)))
    n_rows = int(np.ceil(n_snapshots / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.3*n_cols, 4.3*n_rows))
    axes_flat = np.atleast_1d(axes).flatten()

    for ax, t in zip(axes_flat, snapshots):                        
        plot_arrows_snapshot(ax, cx,cy, wx,wy, assignment, t, R, N_c,
                              n_workers_per_snapshot, cmap_name=cmap_name)
    for ax in axes_flat[len(snapshots):]:        #delating axis for extra axes
        ax.axis("off")

    fig.suptitle(
        f"sample of {n_workers_per_snapshot} workers per snapshot, N_c={N_c}",
        fontsize=12, y=1.0
        )
    plt.show()


plot_arrows_evolution(cx,cy, wx,wy, assignment, N_c, R, n_snapshots=n_snapshots_arrows,
                       n_workers_per_snapshot=n_workers_per_snapshot,
                       cmap_name=arrows_cmap_name, P_max=P_max)'''

def _data_per_point(ax, fig, xlim):
    """
    calculates how many km correspond to 1 typographic point (1/72 inch)
    """
    fig_w_in, _ = fig.get_size_inches()     # get size
    pos = ax.get_position()                 # width, heigth of the axes in figure fraction
    ax_w_in = pos.width * fig_w_in          # width of the axes in inches
    data_w = xlim[1] - xlim[0]              # width of the data in data units (km)
    return (data_w / ax_w_in) / 72.0        


def plot_arrows_snapshot(ax, cx, cy, wx, wy, assignment, t, R, N_c,
                          n_workers_per_snapshot, cmap_name,head_len_pts,
                          head_half_width_pts,shrink_pts,arrow_alpha, arrow_width):
    """
    Vectorized version of per-arrow ax.annotate() loop:
    all arrow shafts are drawn with a single LineCollection and all
    arrowheads with a single PolyCollection, instead of one Annotation
    object per arrow.
    """
    sub_assign = assignment[:t]                    # assignment of the first t workers
    n_show = min(t, n_workers_per_snapshot)        # number of workers to show in the plot
    np.random.seed(0)
    idx = np.random.choice(t, n_show, replace=False)  #choose n_show from t workers  

    cmap = plt.get_cmap(cmap_name)                 
    n_colors = cmap.N if hasattr(cmap, "N") else 10
    center_colors = [cmap(j % n_colors) for j in range(N_c)]

    j_idx = sub_assign[idx]                 # chosen center index per sampled worker
    wx_i, wy_i = wx[idx], wy[idx]           # workers coordinates
    cx_j, cy_j = cx[j_idx], cy[j_idx]       # centers coordinates
    colors = np.array([center_colors[j] for j in j_idx])

    xlim = (-R, R)
    ax.set_xlim(xlim); ax.set_ylim(xlim)   
    ax.set_aspect("equal")

    fig = ax.figure
    data_per_pt = _data_per_point(ax, fig, xlim)   # calling function; how many km correspond to 1 typographic point (1/72 inch)
    head_len = head_len_pts * data_per_pt          # km
    head_hw = head_half_width_pts * data_per_pt
    shrink = shrink_pts * data_per_pt

    # unit direction vector worker -> center for every sampled arrow
    dx, dy = cx_j - wx_i, cy_j - wy_i                 # vector from worker to center; cx_j,... are vectors
    seg_len = np.hypot(dx, dy)                        # vector norm
    seg_len_safe = np.where(seg_len == 0, 1, seg_len) # np.where(condition, value_if_true, value_if_false) preventing division by zero
    ux, uy = dx / seg_len_safe, dy / seg_len_safe     # versors of the arrows

    # shaft endpoints, shrunk slightly at both ends (matches annotate default)
    sx0, sy0 = wx_i + ux * shrink, wy_i + uy * shrink   # starting point of the arrow shaft being a bit away
    sx1, sy1 = cx_j - ux * shrink, cy_j - uy * shrink   # ending point of the arrow shaft

    segments = np.stack([np.column_stack([sx0, sy0]), 
                          np.column_stack([sx1, sy1])], axis=1) # a 3d array (n_show, 2, 2) of the start and end points of each arrow 
    ax.add_collection(LineCollection(segments, colors=colors, linewidths=arrow_width, alpha=arrow_alpha))

    # triangular head at the tip, pointing along (ux, uy)
    tip_x, tip_y = sx1, sy1                                         
    base_cx, base_cy = tip_x - ux * head_len, tip_y - uy * head_len  # base of the triangle
    perp_x, perp_y = -uy, ux

    p1 = np.column_stack([tip_x, tip_y])
    p2 = np.column_stack([base_cx + perp_x * head_hw, base_cy + perp_y * head_hw])
    p3 = np.column_stack([base_cx - perp_x * head_hw, base_cy - perp_y * head_hw])
    tris = np.stack([p1, p2, p3], axis=1)
    ax.add_collection(PolyCollection(tris, facecolors=colors, edgecolors=colors,
                                      alpha=arrow_alpha, linewidths=0))

    ax.add_patch(Circle((0, 0), R, fill=False, linestyle="--", color="gray", linewidth=0.8))
    ax.set_xlim(xlim)
    ax.set_ylim(xlim)
    ax.set_aspect("equal")
    ax.axis('off')
    k_active = int((np.bincount(sub_assign, minlength=N_c) > 0).sum())
    ax.set_title(f"P = {t:,}\nk = {k_active}/{N_c} active centers", fontsize=10)


def plot_arrows_evolution(cx, cy, wx, wy, assignment, N_c, R,
                           n_snapshots, n_workers_per_snapshot,
                           cmap_name, P_init, P_max, 
                           head_len_pts, head_half_width_pts,shrink_pts,
                           arrow_alpha, arrow_width):

    snapshots = np.linspace(P_init, P_max, n_snapshots, dtype=int)   # population at each snapshot

    n_cols = int(np.ceil(np.sqrt(n_snapshots)))
    n_rows = int(np.ceil(n_snapshots / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.3 * n_cols, 4.3 * n_rows))
    axes_flat = np.atleast_1d(axes).flatten()

    for ax, t in zip(axes_flat, snapshots):
        plot_arrows_snapshot(ax, cx, cy, wx, wy, assignment, t, R, N_c,
                              n_workers_per_snapshot, cmap_name, head_len_pts,head_half_width_pts, shrink_pts, arrow_alpha, arrow_width)
    for ax in axes_flat[len(snapshots):]:                 # deleting axis for extra axes
        ax.axis("off")

    fig.suptitle(
        f"sample of {n_workers_per_snapshot} workers per snapshot, N_c={N_c}",
        fontsize=12, y=1.0
    )
    plt.show()

def k_of_P_curve_numpy(assignment):
    '''
    computing cumulative sum array of the number of active centers as a function of the number of workers P,
    given the assignment array
    '''
    _, first_indices = np.unique(assignment, return_index=True) # where each unique index appears for the first time 
    
    activations = np.zeros(len(assignment), dtype=np.uint32)    # initializing
    
    activations[first_indices] = 1            # array of zeros with ones at the first indices of each unique center
    return np.cumsum(activations)

@njit
def k_of_P_curve_numba(assignment,N_c):
    '''
    computing cumulative sum array of the number of active centers as a function of the number of workers P,
    given the assignment array
    '''

    cumsum = np.empty(len(assignment), dtype=np.uint8)     # initializing cumulative sum array
    seen = np.zeros(N_c, dtype=np.bool_)                   # it stores if each center has already appeared

    count = 0                                              # counter

    for i in range(len(assignment)):                             

        assignment_i = assignment[i]                       # scalar of the center assigned to the i-th worker

        if not seen[assignment_i]:                         # if new index
            seen[assignment_i] = True                      # flag it as seen in its first occurrence position 

            count += 1                                     # otherwise it stays the same
            

        cumsum[i] = count                                # cumulative sum, then come back to the loop

    return cumsum