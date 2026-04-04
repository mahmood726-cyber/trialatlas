"""Advanced network statistics engine for TrialAtlas.

Provides centrality measures, degree distribution analysis, modularity
significance testing, spatial autocorrelation (Moran's I), stochastic
block models, ERGM pseudo-likelihood, assortativity, and inequality
metrics (Lorenz/Atkinson).

All stochastic methods accept a seed parameter for reproducibility.
"""

import math
import random
from collections import defaultdict

import numpy as np
from scipy import stats as sp_stats


# ──────────────────────────────────────────────────────────────────────
# Helpers: graph → adjacency structures
# ──────────────────────────────────────────────────────────────────────

def _build_adj(graph):
    """Build adjacency dict and node list from a TrialAtlas graph.

    Returns:
        node_list: list of node names (deterministic order)
        name_to_idx: dict name -> int
        adj: defaultdict(dict) idx -> idx -> weight
        degrees: dict idx -> weighted degree
        total_weight: float sum of all edge weights
    """
    node_list = sorted(graph["nodes"].keys())
    name_to_idx = {name: i for i, name in enumerate(node_list)}
    n = len(node_list)

    adj = defaultdict(lambda: defaultdict(float))
    degrees = defaultdict(float)
    total_weight = 0.0

    for edge in graph["edges"]:
        src = edge["source"]
        tgt = edge["target"]
        w = edge.get("weight", 1)
        if src not in name_to_idx or tgt not in name_to_idx:
            continue
        i = name_to_idx[src]
        j = name_to_idx[tgt]
        adj[i][j] += w
        adj[j][i] += w
        degrees[i] += w
        degrees[j] += w
        total_weight += w

    return node_list, name_to_idx, adj, degrees, total_weight


# ──────────────────────────────────────────────────────────────────────
# 1. Betweenness Centrality (Brandes algorithm)
# ──────────────────────────────────────────────────────────────────────

def betweenness_centrality(graph):
    """Compute betweenness centrality for all nodes using Brandes algorithm.

    Uses BFS (unweighted shortest paths) on the graph. Normalization
    factor is (n-1)(n-2)/2 for undirected graphs.

    Args:
        graph: dict with 'nodes' and 'edges'

    Returns:
        dict {node_name: centrality_value} with values in [0, 1]
    """
    node_list, name_to_idx, adj, degrees, total_weight = _build_adj(graph)
    n = len(node_list)

    if n <= 2:
        return {name: 0.0 for name in node_list}

    cb = [0.0] * n  # centrality accumulator

    for s in range(n):
        # Single-source shortest paths via BFS
        stack = []
        pred = [[] for _ in range(n)]
        sigma = [0.0] * n
        sigma[s] = 1.0
        dist = [-1] * n
        dist[s] = 0
        queue = [s]
        qi = 0

        while qi < len(queue):
            v = queue[qi]
            qi += 1
            stack.append(v)
            for w in adj[v]:
                # First visit
                if dist[w] < 0:
                    dist[w] = dist[v] + 1
                    queue.append(w)
                # Shortest path to w via v?
                if dist[w] == dist[v] + 1:
                    sigma[w] += sigma[v]
                    pred[w].append(v)

        # Back-propagation of dependencies
        delta = [0.0] * n
        while stack:
            w = stack.pop()
            for v in pred[w]:
                delta[v] += (sigma[v] / sigma[w]) * (1.0 + delta[w])
            if w != s:
                cb[w] += delta[w]

    # For undirected graphs, each pair (s,t) is counted from both
    # directions, so divide accumulated values by 2 first.
    cb = [x / 2.0 for x in cb]

    # Normalize: undirected graph factor = (n-1)(n-2)/2
    norm = (n - 1) * (n - 2) / 2.0
    if norm > 0:
        cb = [x / norm for x in cb]

    return {node_list[i]: cb[i] for i in range(n)}


# ──────────────────────────────────────────────────────────────────────
# 2. Eigenvector Centrality (power iteration)
# ──────────────────────────────────────────────────────────────────────

def eigenvector_centrality(graph, max_iter=100, tol=1e-6):
    """Compute eigenvector centrality via power iteration.

    Args:
        graph: dict with 'nodes' and 'edges'
        max_iter: maximum iterations
        tol: convergence tolerance (L2 norm of difference)

    Returns:
        dict {node_name: centrality_value} with values >= 0
    """
    node_list, name_to_idx, adj, degrees, total_weight = _build_adj(graph)
    n = len(node_list)

    if n == 0:
        return {}

    # Initialize x_0 = 1/sqrt(n)
    x = np.full(n, 1.0 / math.sqrt(n))

    for _ in range(max_iter):
        x_new = np.zeros(n)
        for i in range(n):
            for j, w in adj[i].items():
                x_new[i] += w * x[j]

        # Normalize
        norm = np.linalg.norm(x_new)
        if norm > 0:
            x_new /= norm
        else:
            # All zeros -- graph is disconnected or empty edges
            break

        # Check convergence
        if np.linalg.norm(x_new - x) < tol:
            x = x_new
            break
        x = x_new

    # Ensure non-negative (leading eigenvector of non-negative matrix)
    x = np.abs(x)

    return {node_list[i]: float(x[i]) for i in range(n)}


# ──────────────────────────────────────────────────────────────────────
# 3. Degree Distribution Analysis
# ──────────────────────────────────────────────────────────────────────

def degree_distribution(graph):
    """Analyze degree distribution and fit power-law exponent.

    Uses Clauset-Shalizi-Newman MLE for power-law alpha:
        alpha = 1 + n * (sum ln(k_i / k_min))^{-1}

    KS statistic is computed against the fitted power-law CDF.
    is_scale_free = True if KS < critical value (approximate p > 0.1).

    Args:
        graph: dict with 'nodes' and 'edges'

    Returns:
        dict with degrees, frequencies, power_law_alpha, power_law_ks, is_scale_free
    """
    node_list, name_to_idx, adj, degrees_w, total_weight = _build_adj(graph)
    n = len(node_list)

    # Compute unweighted degree for each node
    node_degrees = []
    for i in range(n):
        deg = len(adj[i])
        node_degrees.append(deg)

    if not node_degrees:
        return {
            "degrees": [],
            "frequencies": [],
            "power_law_alpha": float("nan"),
            "power_law_ks": float("nan"),
            "is_scale_free": False,
        }

    # Frequency table
    degree_counts = defaultdict(int)
    for d in node_degrees:
        degree_counts[d] += 1

    sorted_degrees = sorted(degree_counts.keys())
    freq = [degree_counts[d] for d in sorted_degrees]

    # Power-law fit (Clauset-Shalizi-Newman MLE)
    # Filter to degrees >= k_min (use k_min = max(1, min non-zero degree))
    positive_degrees = [d for d in node_degrees if d > 0]

    if len(positive_degrees) < 2:
        return {
            "degrees": sorted_degrees,
            "frequencies": freq,
            "power_law_alpha": float("nan"),
            "power_law_ks": float("nan"),
            "is_scale_free": False,
        }

    k_min = min(positive_degrees)
    filtered = [d for d in positive_degrees if d >= k_min]
    n_fit = len(filtered)

    if n_fit < 2 or k_min == 0:
        alpha = float("nan")
        ks = float("nan")
        is_sf = False
    else:
        # MLE: alpha = 1 + n * (sum ln(k_i / k_min))^{-1}
        # Use k_min - 0.5 (continuous approximation) for discrete data
        sum_log = sum(math.log(k / (k_min - 0.5)) for k in filtered)
        if sum_log > 0:
            alpha = 1.0 + n_fit / sum_log
        else:
            alpha = float("nan")

        # KS statistic: compare empirical CDF with fitted power-law CDF
        if math.isfinite(alpha) and alpha > 1:
            filtered_sorted = sorted(filtered)
            ks = _power_law_ks(filtered_sorted, alpha, k_min)
            # Approximate critical value for p > 0.1
            # Using the standard approximation: c(alpha=0.10) ~ 1.22/sqrt(n)
            critical = 1.22 / math.sqrt(n_fit)
            is_sf = ks < critical
        else:
            ks = float("nan")
            is_sf = False

    return {
        "degrees": sorted_degrees,
        "frequencies": freq,
        "power_law_alpha": alpha,
        "power_law_ks": ks,
        "is_scale_free": is_sf,
    }


def _power_law_ks(data_sorted, alpha, k_min):
    """Compute KS statistic between empirical and power-law CDF."""
    n = len(data_sorted)
    max_d = 0.0
    for i, x in enumerate(data_sorted):
        empirical_cdf = (i + 1) / n
        # Power-law CDF: P(X <= x) = 1 - (k_min / x)^(alpha - 1)
        if x >= k_min and k_min > 0:
            theoretical_cdf = 1.0 - (k_min / x) ** (alpha - 1)
        else:
            theoretical_cdf = 0.0
        d = abs(empirical_cdf - theoretical_cdf)
        if d > max_d:
            max_d = d
    return max_d


# ──────────────────────────────────────────────────────────────────────
# 4. Modularity Significance (permutation test)
# ──────────────────────────────────────────────────────────────────────

def modularity_significance(graph, n_permutations=1000, seed=42):
    """Test whether observed modularity is significantly higher than random.

    Generates random graphs via configuration model (preserving degree
    sequence) and computes modularity for each. Reports z-score and
    empirical p-value.

    Args:
        graph: dict with 'nodes' and 'edges'
        n_permutations: number of random permutations
        seed: random seed

    Returns:
        dict with observed_q, mean_random_q, std_random_q, z_score, p_value
    """
    try:
        from .community_detector import detect_communities
    except ImportError:
        from community_detector import detect_communities

    # Observed modularity
    _, observed_q = detect_communities(graph, seed=seed, return_modularity=True)

    node_list, name_to_idx, adj, degrees, total_weight = _build_adj(graph)
    n = len(node_list)

    if n < 3 or total_weight == 0:
        return {
            "observed_q": observed_q,
            "mean_random_q": 0.0,
            "std_random_q": 0.0,
            "z_score": 0.0,
            "p_value": 1.0,
        }

    # Build degree sequence (stubs)
    # For weighted graphs, use integer edge weights as stub counts
    rng = random.Random(seed)
    edge_list = []
    for edge in graph["edges"]:
        src = edge["source"]
        tgt = edge["target"]
        w = int(edge.get("weight", 1))
        for _ in range(w):
            edge_list.append((src, tgt))

    # Build stubs from degree sequence
    stubs = []
    for edge in graph["edges"]:
        src = edge["source"]
        tgt = edge["target"]
        w = int(edge.get("weight", 1))
        for _ in range(w):
            stubs.append(src)
            stubs.append(tgt)

    random_qs = []
    for _ in range(n_permutations):
        # Configuration model: shuffle stubs and pair them
        shuffled = stubs[:]
        rng.shuffle(shuffled)

        # Build random graph
        random_edges_count = defaultdict(int)
        for k in range(0, len(shuffled) - 1, 2):
            s = shuffled[k]
            t = shuffled[k + 1]
            if s != t:  # no self-loops
                key = (min(s, t), max(s, t))
                random_edges_count[key] += 1

        random_edges = []
        for (s, t), w in random_edges_count.items():
            random_edges.append({"source": s, "target": t, "weight": w})

        random_graph = {"nodes": graph["nodes"], "edges": random_edges}
        _, q = detect_communities(random_graph, seed=seed, return_modularity=True)
        random_qs.append(q)

    mean_q = float(np.mean(random_qs))
    std_q = float(np.std(random_qs, ddof=1)) if len(random_qs) > 1 else 0.0

    if std_q > 0:
        z_score = (observed_q - mean_q) / std_q
    else:
        z_score = 0.0 if observed_q == mean_q else float("inf")

    # Empirical p-value: fraction of random >= observed
    p_value = sum(1 for q in random_qs if q >= observed_q) / len(random_qs)

    return {
        "observed_q": observed_q,
        "mean_random_q": mean_q,
        "std_random_q": std_q,
        "z_score": z_score,
        "p_value": p_value,
    }


# ──────────────────────────────────────────────────────────────────────
# 5. Moran's I (Spatial Autocorrelation)
# ──────────────────────────────────────────────────────────────────────

# Continent mapping for ~20+ major trial countries
COUNTRY_CONTINENTS = {
    "United States": "North America",
    "Canada": "North America",
    "Mexico": "North America",
    "United Kingdom": "Europe",
    "Germany": "Europe",
    "France": "Europe",
    "Italy": "Europe",
    "Spain": "Europe",
    "Netherlands": "Europe",
    "Belgium": "Europe",
    "Switzerland": "Europe",
    "Sweden": "Europe",
    "Denmark": "Europe",
    "Poland": "Europe",
    "Austria": "Europe",
    "Czech Republic": "Europe",
    "China": "Asia",
    "Japan": "Asia",
    "South Korea": "Asia",
    "India": "Asia",
    "Australia": "Oceania",
    "New Zealand": "Oceania",
    "Brazil": "South America",
    "Argentina": "South America",
    "South Africa": "Africa",
    "Egypt": "Africa",
    "Israel": "Middle East",
    "Turkey": "Middle East",
    "Russia": "Europe",
}

# Border adjacency for major trial countries (simplified)
COUNTRY_BORDERS = {
    "United States": {"Canada", "Mexico"},
    "Canada": {"United States"},
    "Mexico": {"United States"},
    "United Kingdom": {"France"},  # Channel Tunnel
    "Germany": {"France", "Netherlands", "Belgium", "Switzerland",
                "Austria", "Poland", "Czech Republic", "Denmark"},
    "France": {"Germany", "Belgium", "Switzerland", "Spain", "Italy",
               "United Kingdom"},
    "Italy": {"France", "Switzerland", "Austria"},
    "Spain": {"France"},
    "Netherlands": {"Germany", "Belgium"},
    "Belgium": {"Germany", "France", "Netherlands"},
    "Switzerland": {"Germany", "France", "Italy", "Austria"},
    "Sweden": {"Denmark"},
    "Denmark": {"Germany", "Sweden"},
    "Poland": {"Germany", "Czech Republic"},
    "Austria": {"Germany", "Switzerland", "Italy", "Czech Republic"},
    "Czech Republic": {"Germany", "Poland", "Austria"},
    "China": {"India", "South Korea", "Russia"},
    "Japan": {"South Korea"},
    "South Korea": {"Japan", "China"},
    "India": {"China"},
    "Australia": {"New Zealand"},
    "New Zealand": {"Australia"},
    "Brazil": {"Argentina"},
    "Argentina": {"Brazil"},
    "South Africa": set(),
    "Egypt": {"Israel"},
    "Israel": {"Egypt", "Turkey"},
    "Turkey": {"Israel", "Russia"},
    "Russia": {"China", "Turkey", "Poland"},
}


def country_adjacency_weights(countries):
    """Build binary spatial weight matrix for a list of countries.

    Returns 1 if two countries share a border OR are on the same continent,
    0 otherwise. Diagonal is always 0.

    Args:
        countries: list of country name strings

    Returns:
        list of lists (n x n weight matrix)
    """
    n = len(countries)
    W = [[0] * n for _ in range(n)]

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            ci = countries[i]
            cj = countries[j]

            # Check border adjacency
            borders_i = COUNTRY_BORDERS.get(ci, set())
            if cj in borders_i:
                W[i][j] = 1
                continue

            # Check same continent
            cont_i = COUNTRY_CONTINENTS.get(ci, "")
            cont_j = COUNTRY_CONTINENTS.get(cj, "")
            if cont_i and cont_j and cont_i == cont_j:
                W[i][j] = 1

    return W


def morans_i(values, weights_matrix):
    """Compute Moran's I for spatial autocorrelation.

    Args:
        values: list of numeric values (one per spatial unit)
        weights_matrix: list of lists (n x n), spatial weight matrix

    Returns:
        dict with I, expected_I, z_score, p_value
    """
    n = len(values)
    if n < 2:
        return {"I": 0.0, "expected_I": 0.0, "z_score": 0.0, "p_value": 1.0}

    x = np.array(values, dtype=float)
    W = np.array(weights_matrix, dtype=float)

    x_bar = np.mean(x)
    dev = x - x_bar
    ss = np.sum(dev ** 2)

    if ss == 0:
        return {"I": 0.0, "expected_I": -1.0 / (n - 1), "z_score": 0.0, "p_value": 1.0}

    # Total weight
    W_sum = np.sum(W)
    if W_sum == 0:
        return {"I": 0.0, "expected_I": -1.0 / (n - 1), "z_score": 0.0, "p_value": 1.0}

    # Moran's I
    numerator = np.sum(W * np.outer(dev, dev))
    I = (n / W_sum) * (numerator / ss)

    # Expected value under null
    E_I = -1.0 / (n - 1)

    # Variance under normality assumption
    S1 = 0.5 * np.sum((W + W.T) ** 2)
    S2 = np.sum((np.sum(W, axis=0) + np.sum(W, axis=1)) ** 2)
    S0 = W_sum

    # Moments of the distribution
    n2 = n * n
    k2 = np.sum(dev ** 4) / n / (ss / n) ** 2  # kurtosis

    A = n * ((n2 - 3 * n + 3) * S1 - n * S2 + 3 * S0 ** 2)
    B = k2 * ((n2 - n) * S1 - 2 * n * S2 + 6 * S0 ** 2)
    C = (n - 1) * (n - 2) * (n - 3) * S0 ** 2

    if C == 0:
        return {"I": float(I), "expected_I": float(E_I), "z_score": 0.0, "p_value": 1.0}

    Var_I = (A - B) / C - E_I ** 2

    if Var_I <= 0:
        return {"I": float(I), "expected_I": float(E_I), "z_score": 0.0, "p_value": 1.0}

    z = (I - E_I) / math.sqrt(Var_I)
    p_value = 2.0 * (1.0 - sp_stats.norm.cdf(abs(z)))

    return {
        "I": float(I),
        "expected_I": float(E_I),
        "z_score": float(z),
        "p_value": float(p_value),
    }


# ──────────────────────────────────────────────────────────────────────
# 6. Stochastic Block Model (Variational EM)
# ──────────────────────────────────────────────────────────────────────

def stochastic_block_model(graph, max_k=5, seed=42):
    """Fit a stochastic block model with variational EM.

    Selects optimal K (number of communities) by ICL criterion.
    Returns soft assignments (probability per community per node).

    Args:
        graph: dict with 'nodes' and 'edges'
        max_k: maximum number of blocks to try (1..max_k)
        seed: random seed

    Returns:
        dict with k_best, assignments, omega_matrix, icl_values
    """
    node_list, name_to_idx, adj, degrees, total_weight = _build_adj(graph)
    n = len(node_list)

    if n == 0:
        return {
            "k_best": 0,
            "assignments": {},
            "omega_matrix": [],
            "icl_values": {},
        }

    # Build binary adjacency matrix
    A = np.zeros((n, n))
    for i in range(n):
        for j, w in adj[i].items():
            A[i][j] = 1.0  # binary for SBM

    rng = np.random.RandomState(seed)
    best_icl = -float("inf")
    best_result = None
    icl_values = {}

    n_restarts = 5  # multiple restarts to avoid degenerate solutions

    for K in range(1, min(max_k + 1, n + 1)):
        best_ll_k = -float("inf")
        best_q_k = None
        best_omega_k = None

        for restart in range(n_restarts):
            # Use spectral initialization for first restart, random for rest
            if restart == 0 and K > 1:
                q_init = _spectral_init(A, K, n, rng)
            else:
                q_init = rng.dirichlet(np.ones(K) * 0.1 + rng.rand(K), size=n)

            q, pi, omega, ll = _sbm_vem(A, K, n, rng, q_init=q_init,
                                         max_iter=100)

            if ll > best_ll_k:
                best_ll_k = ll
                best_q_k = q
                best_omega_k = omega

        # ICL = log-likelihood - penalty
        # Free parameters: (K-1) for pi + K*(K+1)/2 for omega
        # Use BIC-style penalty with log(n) for all parameters
        n_params = (K - 1) + K * (K + 1) / 2.0
        penalty = 0.5 * n_params * math.log(max(n, 2))
        icl = best_ll_k - penalty

        icl_values[K] = float(icl)

        if icl > best_icl:
            best_icl = icl
            best_result = (K, best_q_k, best_omega_k)

    k_best, q_best, omega_best = best_result

    # Build assignments dict
    assignments = {}
    for i in range(n):
        probs = {}
        for k in range(k_best):
            probs[k] = float(q_best[i, k])
        assignments[node_list[i]] = probs

    return {
        "k_best": k_best,
        "assignments": assignments,
        "omega_matrix": omega_best.tolist(),
        "icl_values": icl_values,
    }


def _spectral_init(A, K, n, rng):
    """Spectral initialization for SBM: embed nodes using top-K eigenvectors,
    then assign to clusters via simple k-means-like procedure."""
    # Compute top K eigenvectors of adjacency matrix
    try:
        eigvals, eigvecs = np.linalg.eigh(A)
        # Take top K by magnitude
        idx = np.argsort(np.abs(eigvals))[-K:]
        V = eigvecs[:, idx]  # n x K

        # Simple k-means: assign each node to nearest centroid
        # Initialize centroids from random rows
        centroid_idx = rng.choice(n, size=K, replace=False)
        centroids = V[centroid_idx].copy()

        for _ in range(20):
            # Assign
            labels = np.zeros(n, dtype=int)
            for i in range(n):
                dists = [np.sum((V[i] - centroids[k]) ** 2) for k in range(K)]
                labels[i] = int(np.argmin(dists))

            # Update centroids
            new_centroids = np.zeros_like(centroids)
            for k in range(K):
                members = V[labels == k]
                if len(members) > 0:
                    new_centroids[k] = members.mean(axis=0)
                else:
                    new_centroids[k] = V[rng.randint(n)]
            centroids = new_centroids

        # Convert to soft assignments with high confidence
        q = np.full((n, K), 0.01)
        for i in range(n):
            q[i, labels[i]] = 0.99
        # Normalize rows
        q /= q.sum(axis=1, keepdims=True)
        return q

    except Exception:
        return rng.dirichlet(np.ones(K), size=n)


def _sbm_vem(A, K, n, rng, q_init=None, max_iter=50, tol=1e-6):
    """Variational EM for SBM with K blocks.

    Returns:
        q: n x K matrix of soft assignments
        pi: K vector of block proportions
        omega: K x K matrix of connection probabilities
        ll: variational lower bound (log-likelihood)
    """
    # Initialize q
    if q_init is not None:
        q = q_init.copy()
    else:
        q = rng.dirichlet(np.ones(K), size=n)

    pi = np.mean(q, axis=0)
    omega = np.full((K, K), 0.5)

    eps = 1e-10  # numerical floor

    for iteration in range(max_iter):
        q_old = q.copy()

        # M-step: update pi and omega
        pi = np.mean(q, axis=0)
        pi = np.maximum(pi, eps)
        pi /= pi.sum()

        for k in range(K):
            for l in range(k, K):
                num = 0.0
                den = 0.0
                for i in range(n):
                    for j in range(n):
                        if i == j:
                            continue
                        qij = q[i, k] * q[j, l]
                        if k != l:
                            qij += q[i, l] * q[j, k]
                        num += qij * A[i, j]
                        den += qij
                if den > eps:
                    omega[k, l] = np.clip(num / den, eps, 1.0 - eps)
                else:
                    omega[k, l] = eps
                omega[l, k] = omega[k, l]

        # E-step: update q
        for i in range(n):
            for k in range(K):
                log_q = math.log(pi[k] + eps)
                for j in range(n):
                    if i == j:
                        continue
                    for l in range(K):
                        o_kl = omega[k, l]
                        if A[i, j] > 0:
                            log_q += q[j, l] * math.log(o_kl + eps)
                        else:
                            log_q += q[j, l] * math.log(1.0 - o_kl + eps)
                q[i, k] = log_q

            # Normalize via log-sum-exp
            max_log = max(q[i, :])
            q[i, :] = np.exp(q[i, :] - max_log)
            row_sum = q[i, :].sum()
            if row_sum > 0:
                q[i, :] /= row_sum
            else:
                q[i, :] = 1.0 / K

        # Check convergence
        diff = np.max(np.abs(q - q_old))
        if diff < tol:
            break

    # Compute variational lower bound (approximate log-likelihood)
    ll = 0.0
    for i in range(n):
        for k in range(K):
            if q[i, k] > eps:
                ll += q[i, k] * math.log(pi[k] + eps)
                ll -= q[i, k] * math.log(q[i, k] + eps)
            for j in range(n):
                if i == j:
                    continue
                for l in range(K):
                    qij = q[i, k] * q[j, l]
                    if qij > eps:
                        o_kl = omega[k, l]
                        if A[i, j] > 0:
                            ll += 0.5 * qij * math.log(o_kl + eps)
                        else:
                            ll += 0.5 * qij * math.log(1.0 - o_kl + eps)

    return q, pi, omega, float(ll)


# ──────────────────────────────────────────────────────────────────────
# 7. ERGM Pseudo-Likelihood
# ──────────────────────────────────────────────────────────────────────

def ergm_pseudolikelihood(graph, features_func=None):
    """Fit an ERGM via maximum pseudo-likelihood (logistic regression on dyads).

    Default features per potential edge (i, j):
        - edges: constant 1 (intercept/baseline)
        - sponsor_homophily: 1 if both nodes are same type
        - shared_conditions: count of shared conditions
        - same_country: 1 if nodes share a country
        - degree_sum: sum of degrees of endpoints

    Args:
        graph: dict with 'nodes' and 'edges'
        features_func: optional callable(graph, i, j) -> list of float
                       If None, uses default features.

    Returns:
        dict with coefficients (list of dicts) and pseudo_r2
    """
    from sklearn.linear_model import LogisticRegression

    node_list, name_to_idx, adj, degrees, total_weight = _build_adj(graph)
    n = len(node_list)
    nodes_meta = graph["nodes"]

    if n < 3:
        return {"coefficients": [], "pseudo_r2": 0.0}

    # Build edge set for quick lookup
    edge_set = set()
    for edge in graph["edges"]:
        s = edge["source"]
        t = edge["target"]
        edge_set.add((s, t))
        edge_set.add((t, s))

    feature_names = ["edges", "same_type", "shared_conditions",
                     "same_country", "degree_sum"]

    # Compute features for all dyads
    X_rows = []
    y_rows = []

    for i in range(n):
        for j in range(i + 1, n):
            ni = node_list[i]
            nj = node_list[j]

            if features_func is not None:
                feat = features_func(graph, ni, nj)
            else:
                feat = _default_ergm_features(
                    ni, nj, nodes_meta, degrees, name_to_idx
                )

            X_rows.append(feat)
            y_val = 1.0 if (ni, nj) in edge_set else 0.0
            y_rows.append(y_val)

    X = np.array(X_rows)
    y = np.array(y_rows)

    # Check we have both classes
    if len(set(y)) < 2:
        return {
            "coefficients": [
                {"name": fn, "coef": 0.0, "se": 0.0, "z": 0.0, "p_value": 1.0}
                for fn in feature_names
            ],
            "pseudo_r2": 0.0,
        }

    # Fit logistic regression (no penalty for interpretable coefficients)
    # Use large C for minimal regularization
    model = LogisticRegression(
        penalty="l2", C=1e6, max_iter=1000, solver="lbfgs", fit_intercept=False
    )
    model.fit(X, y)

    coefs = model.coef_[0]

    # Standard errors from Hessian (Fisher information)
    # P(y=1|x) for each observation
    probs = model.predict_proba(X)[:, 1]
    W_diag = probs * (1 - probs)
    W_diag = np.maximum(W_diag, 1e-10)

    # Fisher information: X^T W X
    XtWX = X.T @ np.diag(W_diag) @ X
    try:
        cov = np.linalg.inv(XtWX)
        se = np.sqrt(np.maximum(np.diag(cov), 0))
    except np.linalg.LinAlgError:
        se = np.zeros(len(coefs))

    # Z-scores and p-values
    coefficients = []
    for k, fn in enumerate(feature_names):
        z = coefs[k] / se[k] if se[k] > 0 else 0.0
        p = 2.0 * (1.0 - sp_stats.norm.cdf(abs(z)))
        coefficients.append({
            "name": fn,
            "coef": float(coefs[k]),
            "se": float(se[k]),
            "z": float(z),
            "p_value": float(p),
        })

    # McFadden's pseudo R^2
    ll_model = np.sum(
        y * np.log(np.maximum(probs, 1e-10))
        + (1 - y) * np.log(np.maximum(1 - probs, 1e-10))
    )
    p_null = np.mean(y)
    if p_null > 0 and p_null < 1:
        ll_null = len(y) * (
            p_null * math.log(p_null)
            + (1 - p_null) * math.log(1 - p_null)
        )
    else:
        ll_null = -1e-10

    pseudo_r2 = 1.0 - (ll_model / ll_null) if ll_null != 0 else 0.0
    pseudo_r2 = max(0.0, min(1.0, pseudo_r2))

    return {
        "coefficients": coefficients,
        "pseudo_r2": float(pseudo_r2),
    }


def _default_ergm_features(ni, nj, nodes_meta, degrees, name_to_idx):
    """Compute default ERGM features for a dyad (ni, nj)."""
    meta_i = nodes_meta.get(ni, {})
    meta_j = nodes_meta.get(nj, {})

    # 1. Edges (constant)
    feat_edges = 1.0

    # 2. Same type (sponsor homophily)
    type_i = meta_i.get("type", "")
    type_j = meta_j.get("type", "")
    feat_same_type = 1.0 if type_i == type_j else 0.0

    # 3. Shared conditions
    conds_i = set(meta_i.get("conditions", []))
    conds_j = set(meta_j.get("conditions", []))
    feat_shared = float(len(conds_i & conds_j))

    # 4. Same country
    country_i = meta_i.get("country", "")
    country_j = meta_j.get("country", "")
    # country can be a string or a list
    if isinstance(country_i, list):
        countries_i = set(country_i)
    else:
        countries_i = {country_i} if country_i else set()
    if isinstance(country_j, list):
        countries_j = set(country_j)
    else:
        countries_j = {country_j} if country_j else set()
    feat_same_country = 1.0 if countries_i & countries_j else 0.0

    # 5. Degree sum
    idx_i = name_to_idx.get(ni, 0)
    idx_j = name_to_idx.get(nj, 0)
    feat_degree_sum = degrees.get(idx_i, 0) + degrees.get(idx_j, 0)

    return [feat_edges, feat_same_type, feat_shared, feat_same_country,
            feat_degree_sum]


# ──────────────────────────────────────────────────────────────────────
# 8. Assortativity Coefficients
# ──────────────────────────────────────────────────────────────────────

def degree_assortativity(graph):
    """Compute degree assortativity coefficient.

    r = (M * sum(j_e * k_e) - [sum(j_e + k_e)/2]^2)
        / (M * sum(j_e^2 + k_e^2)/2 - [sum(j_e + k_e)/2]^2)

    Where j_e, k_e are degrees at each end of edge e, M = number of edges.

    Args:
        graph: dict with 'nodes' and 'edges'

    Returns:
        float in [-1, 1], or 0.0 if undefined
    """
    node_list, name_to_idx, adj, degrees_w, total_weight = _build_adj(graph)

    # Compute unweighted degree
    deg = defaultdict(int)
    for i in range(len(node_list)):
        deg[i] = len(adj[i])

    edges = graph["edges"]
    M = len(edges)

    if M == 0:
        return 0.0

    sum_jk = 0.0      # sum of j_e * k_e
    sum_jk_sq = 0.0   # sum of (j_e^2 + k_e^2)
    sum_jpk = 0.0     # sum of (j_e + k_e)

    for edge in edges:
        src = edge["source"]
        tgt = edge["target"]
        if src not in name_to_idx or tgt not in name_to_idx:
            continue
        j_e = deg[name_to_idx[src]]
        k_e = deg[name_to_idx[tgt]]
        sum_jk += j_e * k_e
        sum_jk_sq += j_e ** 2 + k_e ** 2
        sum_jpk += j_e + k_e

    numerator = M * sum_jk - (sum_jpk / 2.0) ** 2
    denominator = M * sum_jk_sq / 2.0 - (sum_jpk / 2.0) ** 2

    if abs(denominator) < 1e-15:
        return 0.0

    r = numerator / denominator
    return float(np.clip(r, -1.0, 1.0))


def attribute_assortativity(graph, attribute):
    """Compute attribute assortativity (categorical mixing).

    Fraction of edges connecting same-attribute nodes vs expected under
    random mixing (Newman's assortativity for categorical attributes).

    r = (trace(e) - ||e^2||) / (1 - ||e^2||)

    where e is the mixing matrix normalized by total edges.

    Args:
        graph: dict with 'nodes' and 'edges'
        attribute: str, node metadata key to use (e.g., 'type', 'country')

    Returns:
        float in [-1, 1], or 0.0 if undefined
    """
    nodes_meta = graph["nodes"]
    edges = graph["edges"]

    if not edges:
        return 0.0

    # Get attribute values
    def _get_attr(node_name):
        meta = nodes_meta.get(node_name, {})
        val = meta.get(attribute, "unknown")
        if isinstance(val, list):
            return val[0] if val else "unknown"
        return str(val)

    # Collect unique attribute values
    attr_values = set()
    for name in nodes_meta:
        attr_values.add(_get_attr(name))
    attr_list = sorted(attr_values)
    attr_to_idx = {a: i for i, a in enumerate(attr_list)}
    k = len(attr_list)

    if k < 2:
        return 0.0

    # Build mixing matrix
    e = np.zeros((k, k))
    for edge in edges:
        ai = attr_to_idx[_get_attr(edge["source"])]
        aj = attr_to_idx[_get_attr(edge["target"])]
        e[ai, aj] += 1
        e[aj, ai] += 1

    total = e.sum()
    if total == 0:
        return 0.0

    e /= total

    # r = (trace(e) - sum(a_i^2)) / (1 - sum(a_i^2))
    # where a_i = sum_j e_ij (row/column sums)
    a = e.sum(axis=1)
    trace_e = np.trace(e)
    sum_a2 = np.sum(a ** 2)

    denom = 1.0 - sum_a2
    if abs(denom) < 1e-15:
        return 0.0

    r = (trace_e - sum_a2) / denom
    return float(np.clip(r, -1.0, 1.0))


# ──────────────────────────────────────────────────────────────────────
# 9. Lorenz Curve + Atkinson Index
# ──────────────────────────────────────────────────────────────────────

def lorenz_curve(values):
    """Compute Lorenz curve and Gini coefficient.

    Args:
        values: list of non-negative numeric values

    Returns:
        dict with percentiles, cumulative_shares, gini
    """
    if not values:
        return {"percentiles": [], "cumulative_shares": [], "gini": 0.0}

    vals = sorted(values)
    n = len(vals)
    total = sum(vals)

    if total == 0:
        percentiles = [(i + 1) / n for i in range(n)]
        cumulative_shares = [1.0 / n * (i + 1) for i in range(n)]
        return {
            "percentiles": percentiles,
            "cumulative_shares": cumulative_shares,
            "gini": 0.0,
        }

    cumsum = 0.0
    percentiles = []
    cumulative_shares = []
    for i, v in enumerate(vals):
        cumsum += v
        percentiles.append((i + 1) / n)
        cumulative_shares.append(cumsum / total)

    # Gini via trapezoidal rule on Lorenz curve
    # G = 1 - 2 * Area under Lorenz curve
    # Area = sum of trapezoids from (0,0) to each point
    area = 0.0
    prev_p = 0.0
    prev_cs = 0.0
    for p, cs in zip(percentiles, cumulative_shares):
        area += 0.5 * (p - prev_p) * (cs + prev_cs)
        prev_p = p
        prev_cs = cs

    gini = 1.0 - 2.0 * area
    gini = max(0.0, min(1.0, gini))

    return {
        "percentiles": percentiles,
        "cumulative_shares": cumulative_shares,
        "gini": gini,
    }


def atkinson_index(values, epsilon=0.5):
    """Compute the Atkinson inequality index.

    For epsilon != 1:
        A = 1 - (1/n * sum((x_i/mu)^(1-epsilon)))^(1/(1-epsilon))

    For epsilon == 1:
        A = 1 - (prod(x_i))^(1/n) / mu  (geometric mean / arithmetic mean)

    Args:
        values: list of positive numeric values
        epsilon: inequality aversion parameter (0=indifferent, higher=more averse)

    Returns:
        float in [0, 1]
    """
    if not values:
        return 0.0

    vals = [max(v, 1e-10) for v in values]  # floor at small positive
    n = len(vals)
    mu = sum(vals) / n

    if mu <= 0:
        return 0.0

    if abs(epsilon - 1.0) < 1e-10:
        # Geometric mean / arithmetic mean
        log_sum = sum(math.log(v) for v in vals)
        geo_mean = math.exp(log_sum / n)
        A = 1.0 - geo_mean / mu
    else:
        # General formula
        exponent = 1.0 - epsilon
        mean_ratio = sum((v / mu) ** exponent for v in vals) / n
        if mean_ratio > 0:
            A = 1.0 - mean_ratio ** (1.0 / exponent)
        else:
            A = 1.0

    return float(np.clip(A, 0.0, 1.0))


# ──────────────────────────────────────────────────────────────────────
# 12. Spectral Gap Analysis
# ──────────────────────────────────────────────────────────────────────

def spectral_gap(graph, max_k=10):
    """Compute spectral gap analysis of the graph Laplacian.

    Eigenvalues of L = D - A reveal cluster structure:
    - Number of zero eigenvalues = number of connected components
    - Largest gap in the first max_k eigenvalues suggests optimal k for clustering
    - Fiedler value (second eigenvalue) measures algebraic connectivity

    Args:
        graph: dict with 'nodes' and 'edges'
        max_k: maximum k to consider for gap analysis

    Returns:
        dict with eigenvalues, gaps, suggested_k, fiedler_value
    """
    node_list, name_to_idx, adj, degrees, total_weight = _build_adj(graph)
    n = len(node_list)

    if n == 0:
        return {"eigenvalues": [], "gaps": [], "suggested_k": 1, "fiedler_value": 0.0}

    # Build adjacency matrix A and degree matrix D
    A = np.zeros((n, n), dtype=float)
    for i in adj:
        for j, w in adj[i].items():
            A[i, j] = w

    D = np.diag(A.sum(axis=1))
    L = D - A

    # Compute eigenvalues (symmetric → eigvalsh is stable)
    eigenvalues = sorted(np.linalg.eigvalsh(L).tolist())

    # Compute gaps
    gaps = [eigenvalues[i + 1] - eigenvalues[i] for i in range(len(eigenvalues) - 1)]

    # Suggested k: argmax of gaps in first max_k gaps
    search_range = min(max_k, len(gaps))
    if search_range > 0:
        suggested_k = int(np.argmax(gaps[:search_range])) + 1
    else:
        suggested_k = 1

    # Fiedler value: second smallest eigenvalue (index 1 if n >= 2)
    fiedler_value = eigenvalues[1] if n >= 2 else 0.0

    return {
        "eigenvalues": eigenvalues,
        "gaps": gaps,
        "suggested_k": suggested_k,
        "fiedler_value": float(fiedler_value),
    }


# ──────────────────────────────────────────────────────────────────────
# 13. Network Motif Census
# ──────────────────────────────────────────────────────────────────────

def motif_census(graph, n_random=100, seed=42):
    """Count 3-node subgraph motifs (triads) and compute significance.

    Triad types for undirected graphs:
    - "empty": no edges among the 3 nodes
    - "edge": exactly 1 edge (open pair)
    - "wedge": exactly 2 edges (V-shape / path)
    - "triangle": all 3 edges present

    Significance is computed as z-score against configuration-model
    random graphs.

    Args:
        graph: dict with 'nodes' and 'edges'
        n_random: number of random graphs for significance testing
        seed: RNG seed for reproducibility

    Returns:
        dict with triads, triad_significance, total_triads
    """
    rng = random.Random(seed)
    node_list, name_to_idx, adj, degrees, total_weight = _build_adj(graph)
    n = len(node_list)

    # Build adjacency set for fast lookup
    edge_set = set()
    for i in adj:
        for j in adj[i]:
            if i < j:
                edge_set.add((i, j))

    def _count_triads(es, n_nodes):
        """Count triad types for a given edge set."""
        # Build adjacency list from edge set
        nbrs = defaultdict(set)
        for i, j in es:
            nbrs[i].add(j)
            nbrs[j].add(i)

        # Count wedges (pairs of neighbors) and triangles
        total_triangles = 0
        total_wedges = 0
        for u in range(n_nodes):
            nbrs_u = sorted(nbrs[u])
            k = len(nbrs_u)
            # Number of wedges centered at u
            total_wedges += k * (k - 1) // 2
            # Count triangles involving u
            for idx_a in range(k):
                for idx_b in range(idx_a + 1, k):
                    a, b = nbrs_u[idx_a], nbrs_u[idx_b]
                    if b in nbrs[a]:
                        total_triangles += 1

        # Each triangle counted 3 times (once per vertex)
        total_triangles //= 3
        # Each triangle contributes 3 wedges
        total_wedges -= total_triangles * 3  # open wedges only
        # total_wedges now = open wedges (V-shapes without closing edge)

        return {"triangle": total_triangles, "wedge": total_wedges}

    # Count in observed graph
    observed = _count_triads(edge_set, n)
    total_triads = observed["triangle"] + observed["wedge"]

    # Configuration model random graphs for significance
    edge_list = list(edge_set)
    m = len(edge_list)
    degree_seq = defaultdict(int)
    for i, j in edge_list:
        degree_seq[i] += 1
        degree_seq[j] += 1

    random_counts = {"triangle": [], "wedge": []}

    for _ in range(n_random):
        # Configuration model: create random graph preserving degree sequence
        stubs = []
        for node, deg in degree_seq.items():
            stubs.extend([node] * deg)
        rng.shuffle(stubs)

        rand_edges = set()
        for k in range(0, len(stubs) - 1, 2):
            u, v = stubs[k], stubs[k + 1]
            if u != v:  # no self-loops
                rand_edges.add((min(u, v), max(u, v)))

        rc = _count_triads(rand_edges, n)
        random_counts["triangle"].append(rc["triangle"])
        random_counts["wedge"].append(rc["wedge"])

    # Compute z-scores
    triad_significance = {}
    for motif_type in ["triangle", "wedge"]:
        rand_vals = random_counts[motif_type]
        mean_r = np.mean(rand_vals)
        std_r = np.std(rand_vals, ddof=0)
        if std_r > 0:
            z = (observed[motif_type] - mean_r) / std_r
        else:
            z = 0.0
        triad_significance[motif_type] = float(z)

    return {
        "triads": {k: v for k, v in observed.items()},
        "triad_significance": triad_significance,
        "total_triads": total_triads,
    }


# ──────────────────────────────────────────────────────────────────────
# 14. Link Prediction
# ──────────────────────────────────────────────────────────────────────

def link_prediction(graph, test_fraction=0.2, seed=42):
    """Predict missing/future edges using neighborhood-based scores.

    Methods:
    - Common Neighbors: |N(u) ∩ N(v)|
    - Jaccard coefficient: |N(u) ∩ N(v)| / |N(u) ∪ N(v)|
    - Adamic-Adar: sum(1/log(|N(w)|)) for w in N(u) ∩ N(v)

    Evaluation: hold out test_fraction of edges, predict, compute AUC.

    Args:
        graph: dict with 'nodes' and 'edges'
        test_fraction: fraction of edges to hold out for evaluation
        seed: RNG seed for reproducibility

    Returns:
        dict with predictions, AUC per method, top_predicted
    """
    rng = random.Random(seed)
    node_list, name_to_idx, adj, degrees, total_weight = _build_adj(graph)
    n = len(node_list)

    # Build edge set
    all_edges = set()
    for i in adj:
        for j in adj[i]:
            if i < j:
                all_edges.add((i, j))

    edge_list = sorted(all_edges)

    # Split into train and test
    n_test = max(1, int(len(edge_list) * test_fraction))
    if test_fraction <= 0 or len(edge_list) < 3:
        n_test = 0

    shuffled = list(edge_list)
    rng.shuffle(shuffled)
    test_edges = set(shuffled[:n_test])
    train_edges = all_edges - test_edges

    # Build training adjacency
    train_nbrs = defaultdict(set)
    for i, j in train_edges:
        train_nbrs[i].add(j)
        train_nbrs[j].add(i)

    def _common_neighbors(u, v):
        return len(train_nbrs[u] & train_nbrs[v])

    def _jaccard(u, v):
        inter = len(train_nbrs[u] & train_nbrs[v])
        union = len(train_nbrs[u] | train_nbrs[v])
        return inter / union if union > 0 else 0.0

    def _adamic_adar(u, v):
        score = 0.0
        for w in train_nbrs[u] & train_nbrs[v]:
            deg_w = len(train_nbrs[w])
            if deg_w > 1:
                score += 1.0 / math.log(deg_w)
        return score

    # Score all non-edges (in training graph)
    non_edges = []
    # For efficiency, limit to node pairs within distance 2
    candidates = set()
    for u in range(n):
        for v in train_nbrs[u]:
            for w in train_nbrs[v]:
                if w > u and (min(u, w), max(u, w)) not in train_edges:
                    candidates.add((u, w))

    predictions = []
    for u, v in candidates:
        cn = _common_neighbors(u, v)
        jc = _jaccard(u, v)
        aa = _adamic_adar(u, v)
        predictions.append({
            "source": node_list[u],
            "target": node_list[v],
            "common_neighbors": cn,
            "jaccard": jc,
            "adamic_adar": aa,
        })

    # Compute AUC for each method if we have test edges
    auc_cn = 0.5
    auc_jc = 0.5
    auc_aa = 0.5

    if n_test > 0 and len(candidates) > 0:
        # Build score dicts
        score_map_cn = {}
        score_map_jc = {}
        score_map_aa = {}
        for p in predictions:
            key = (name_to_idx[p["source"]], name_to_idx[p["target"]])
            score_map_cn[key] = p["common_neighbors"]
            score_map_jc[key] = p["jaccard"]
            score_map_aa[key] = p["adamic_adar"]

        # For AUC: compare test edges vs random non-edges
        test_edge_list = list(test_edges)
        non_test_non_edges = [e for e in candidates if e not in test_edges]

        if len(non_test_non_edges) > 0 and len(test_edge_list) > 0:
            n_comparisons = min(1000, len(test_edge_list) * len(non_test_non_edges))
            wins_cn, wins_jc, wins_aa = 0, 0, 0
            total_comp = 0

            for _ in range(n_comparisons):
                pos = test_edge_list[rng.randint(0, len(test_edge_list) - 1)]
                neg = non_test_non_edges[rng.randint(0, len(non_test_non_edges) - 1)]

                pos_key = (min(pos[0], pos[1]), max(pos[0], pos[1]))
                neg_key = (min(neg[0], neg[1]), max(neg[0], neg[1]))

                s_pos_cn = score_map_cn.get(pos_key, 0)
                s_neg_cn = score_map_cn.get(neg_key, 0)
                s_pos_jc = score_map_jc.get(pos_key, 0)
                s_neg_jc = score_map_jc.get(neg_key, 0)
                s_pos_aa = score_map_aa.get(pos_key, 0)
                s_neg_aa = score_map_aa.get(neg_key, 0)

                if s_pos_cn > s_neg_cn:
                    wins_cn += 1
                elif s_pos_cn == s_neg_cn:
                    wins_cn += 0.5
                if s_pos_jc > s_neg_jc:
                    wins_jc += 1
                elif s_pos_jc == s_neg_jc:
                    wins_jc += 0.5
                if s_pos_aa > s_neg_aa:
                    wins_aa += 1
                elif s_pos_aa == s_neg_aa:
                    wins_aa += 0.5

                total_comp += 1

            if total_comp > 0:
                auc_cn = wins_cn / total_comp
                auc_jc = wins_jc / total_comp
                auc_aa = wins_aa / total_comp

    # Sort predictions by Adamic-Adar score
    predictions.sort(key=lambda p: p["adamic_adar"], reverse=True)
    top_predicted = [
        {"source": p["source"], "target": p["target"], "score": p["adamic_adar"]}
        for p in predictions[:10]
    ]

    return {
        "predictions": predictions,
        "auc_common_neighbors": float(auc_cn),
        "auc_jaccard": float(auc_jc),
        "auc_adamic_adar": float(auc_aa),
        "top_predicted": top_predicted,
    }


# ──────────────────────────────────────────────────────────────────────
# 15. Core-Periphery Detection (Borgatti-Everett)
# ──────────────────────────────────────────────────────────────────────

def core_periphery(graph, seed=42):
    """Detect core-periphery structure using Borgatti-Everett model.

    The ideal core-periphery pattern:
    - Core-core: all edges present
    - Core-periphery: all edges present
    - Periphery-periphery: no edges

    Uses greedy hill-climbing with random restarts to maximize
    Pearson correlation between adjacency matrix and ideal pattern.

    Args:
        graph: dict with 'nodes' and 'edges'
        seed: RNG seed for reproducibility

    Returns:
        dict with core_nodes, periphery_nodes, core_quality, correlation
    """
    rng = random.Random(seed)
    node_list, name_to_idx, adj, degrees, total_weight = _build_adj(graph)
    n = len(node_list)

    if n == 0:
        return {
            "core_nodes": [],
            "periphery_nodes": [],
            "core_quality": 0.0,
            "correlation": 0.0,
        }

    # Build adjacency matrix (binary for correlation)
    A = np.zeros((n, n), dtype=float)
    for i in adj:
        for j, w in adj[i].items():
            A[i, j] = 1.0  # binary

    def _compute_correlation(assignment):
        """Compute Pearson correlation between A and ideal pattern."""
        ideal = np.zeros((n, n), dtype=float)
        for i in range(n):
            for j in range(n):
                if i != j:
                    ideal[i, j] = max(assignment[i], assignment[j])

        # Flatten upper triangle only
        idx = np.triu_indices(n, k=1)
        a_flat = A[idx]
        i_flat = ideal[idx]

        if np.std(a_flat) < 1e-12 or np.std(i_flat) < 1e-12:
            return 0.0

        return float(np.corrcoef(a_flat, i_flat)[0, 1])

    best_assignment = None
    best_corr = -2.0

    # Multiple random restarts
    n_restarts = 10
    for _ in range(n_restarts):
        # Initialize: assign top-degree nodes to core
        assignment = [0] * n
        n_core = max(1, n // 3)

        # Random initialization
        core_indices = rng.sample(range(n), min(n_core, n))
        for idx in core_indices:
            assignment[idx] = 1

        # Greedy hill-climbing
        improved = True
        current_corr = _compute_correlation(assignment)
        while improved:
            improved = False
            for i in range(n):
                # Try flipping node i
                assignment[i] = 1 - assignment[i]
                new_corr = _compute_correlation(assignment)

                if new_corr > current_corr:
                    current_corr = new_corr
                    improved = True
                else:
                    assignment[i] = 1 - assignment[i]  # flip back

        corr = _compute_correlation(assignment)
        if corr > best_corr:
            best_corr = corr
            best_assignment = list(assignment)

    # Extract results
    core_nodes = [node_list[i] for i in range(n) if best_assignment[i] == 1]
    periphery_nodes = [node_list[i] for i in range(n) if best_assignment[i] == 0]

    # Core quality: density of core subgraph
    n_core = len(core_nodes)
    if n_core >= 2:
        core_idx = [i for i in range(n) if best_assignment[i] == 1]
        core_edges = sum(1 for i in core_idx for j in core_idx if i < j and A[i, j] > 0)
        max_core_edges = n_core * (n_core - 1) // 2
        core_quality = core_edges / max_core_edges if max_core_edges > 0 else 0.0
    else:
        core_quality = 0.0

    return {
        "core_nodes": core_nodes,
        "periphery_nodes": periphery_nodes,
        "core_quality": float(core_quality),
        "correlation": float(best_corr),
    }


# ──────────────────────────────────────────────────────────────────────
# 16. Network Robustness (Percolation Analysis)
# ──────────────────────────────────────────────────────────────────────

def network_robustness(graph, strategy='targeted', seed=42):
    """Analyze network robustness via node removal percolation.

    Remove nodes progressively and track the size of the largest
    connected component (giant component).

    Strategies:
    - 'targeted': remove highest-degree nodes first (worst-case attack)
    - 'random': remove nodes in random order (random failure)

    Args:
        graph: dict with 'nodes' and 'edges'
        strategy: 'targeted' or 'random'
        seed: RNG seed for reproducibility (used for 'random' strategy)

    Returns:
        dict with removal_fractions, giant_component_sizes,
        critical_threshold, robustness_index
    """
    rng = random.Random(seed)
    node_list, name_to_idx, adj, degrees, total_weight = _build_adj(graph)
    n = len(node_list)

    if n == 0:
        return {
            "removal_fractions": [0.0],
            "giant_component_sizes": [0],
            "critical_threshold": 0.0,
            "robustness_index": 0.0,
        }

    # Determine removal order
    if strategy == 'targeted':
        # Sort by degree descending
        order = sorted(range(n), key=lambda i: degrees.get(i, 0), reverse=True)
    else:
        order = list(range(n))
        rng.shuffle(order)

    # Fractions to evaluate
    fractions = [round(f * 0.05, 2) for f in range(21)]  # 0.0, 0.05, ..., 1.0

    removal_fractions = []
    giant_component_sizes = []
    removed = set()

    def _giant_component_size(removed_set):
        """BFS to find largest connected component."""
        visited = set()
        max_size = 0
        for start in range(n):
            if start in removed_set or start in visited:
                continue
            # BFS
            queue = [start]
            visited.add(start)
            size = 0
            while queue:
                node = queue.pop(0)
                size += 1
                for nbr in adj.get(node, {}):
                    if nbr not in removed_set and nbr not in visited:
                        visited.add(nbr)
                        queue.append(nbr)
            max_size = max(max_size, size)
        return max_size

    order_idx = 0
    for frac in fractions:
        n_remove = int(round(frac * n))
        # Remove nodes up to n_remove
        while len(removed) < n_remove and order_idx < n:
            removed.add(order[order_idx])
            order_idx += 1

        gc_size = _giant_component_size(removed)
        removal_fractions.append(frac)
        giant_component_sizes.append(gc_size)

    # Normalize giant component sizes by original size
    gc_normalized = [s / n for s in giant_component_sizes]

    # Critical threshold: first fraction where giant component < 50% of original
    critical_threshold = 1.0
    for i, (frac, gc_n) in enumerate(zip(removal_fractions, gc_normalized)):
        if gc_n < 0.5:
            critical_threshold = frac
            break

    # Robustness index: area under the curve (trapezoidal integration)
    _trapz = getattr(np, 'trapezoid', np.trapz)
    robustness_index = float(_trapz(gc_normalized, removal_fractions))

    return {
        "removal_fractions": removal_fractions,
        "giant_component_sizes": giant_component_sizes,
        "critical_threshold": float(critical_threshold),
        "robustness_index": float(robustness_index),
    }


# ──────────────────────────────────────────────────────────────────────
# 17. Spectral Embedding (Normalized Laplacian eigendecomposition)
# ──────────────────────────────────────────────────────────────────────

def spectral_embedding(graph, n_dims=2):
    """Embed graph nodes via eigendecomposition of normalized Laplacian.

    Computes L_sym = D^{-1/2} L D^{-1/2} and takes the first n_dims
    non-trivial eigenvectors (those after the zero eigenvalue) as
    coordinate axes.

    Args:
        graph: TrialAtlas graph dict with "nodes" and "edges".
        n_dims: Number of embedding dimensions (default 2).

    Returns:
        dict with:
            embeddings: {node_name: [coord_1, ..., coord_n_dims]}
            explained_variance: list of eigenvalues used (fraction of total).
    """
    node_list, name_to_idx, adj, degrees, total_weight = _build_adj(graph)
    n = len(node_list)

    if n < 2:
        return {
            "embeddings": {node_list[0]: [0.0] * n_dims} if n == 1 else {},
            "explained_variance": [0.0] * n_dims,
        }

    # Build adjacency matrix A and degree matrix D
    A = np.zeros((n, n))
    for i in adj:
        for j, w in adj[i].items():
            A[i][j] = w

    D_diag = np.array([degrees.get(i, 0.0) for i in range(n)])
    # Avoid division by zero for isolated nodes
    D_inv_sqrt = np.zeros(n)
    for i in range(n):
        if D_diag[i] > 0:
            D_inv_sqrt[i] = 1.0 / math.sqrt(D_diag[i])

    # Laplacian L = D - A
    L = np.diag(D_diag) - A

    # Normalized Laplacian L_sym = D^{-1/2} L D^{-1/2}
    D_inv_sqrt_mat = np.diag(D_inv_sqrt)
    L_sym = D_inv_sqrt_mat @ L @ D_inv_sqrt_mat

    # Symmetrize to avoid numerical issues
    L_sym = (L_sym + L_sym.T) / 2.0

    eigenvalues, eigenvectors = np.linalg.eigh(L_sym)

    # Skip first eigenvector (trivial, eigenvalue ~0)
    # Take next n_dims eigenvectors
    actual_dims = min(n_dims, n - 1)
    selected_eigenvalues = eigenvalues[1:1 + actual_dims]
    selected_vectors = eigenvectors[:, 1:1 + actual_dims]

    # Explained variance: fraction of total eigenvalue sum
    total_eigensum = float(np.sum(eigenvalues[1:]))  # exclude trivial
    if total_eigensum > 0:
        explained = [float(ev / total_eigensum) for ev in selected_eigenvalues]
    else:
        explained = [0.0] * actual_dims

    # Pad if n_dims > actual_dims
    if actual_dims < n_dims:
        explained += [0.0] * (n_dims - actual_dims)

    embeddings = {}
    for i, name in enumerate(node_list):
        coords = []
        for d in range(actual_dims):
            coords.append(float(selected_vectors[i, d]))
        # Pad
        coords += [0.0] * (n_dims - actual_dims)
        embeddings[name] = coords

    return {
        "embeddings": embeddings,
        "explained_variance": explained,
    }


# ──────────────────────────────────────────────────────────────────────
# 18. Temporal Network Analysis
# ──────────────────────────────────────────────────────────────────────

def temporal_network_analysis(trials_with_dates):
    """Analyze temporal evolution of a trial collaboration network.

    Builds yearly snapshots tracking growth, preferential attachment,
    and densification.

    Args:
        trials_with_dates: list of dicts, each with:
            nctId, sponsor, year (int), locations: [{facility, country}]

    Returns:
        dict with:
            snapshots: list of {year, n_nodes, n_edges, density}
            growth_rate: average annual node growth rate
            preferential_attachment_alpha: exponent alpha where
                P(edge to v) ~ deg(v)^alpha
            densification_exponent: beta where e ~ n^beta
    """
    # Sort trials by year
    sorted_trials = sorted(trials_with_dates, key=lambda t: t.get("year", 0))
    if not sorted_trials:
        return {
            "snapshots": [],
            "growth_rate": 0.0,
            "preferential_attachment_alpha": 0.0,
            "densification_exponent": 0.0,
        }

    # Build cumulative snapshots per year
    all_nodes = set()
    all_edges = set()
    years_seen = []
    snapshots = []

    # Group by year
    year_groups = defaultdict(list)
    for trial in sorted_trials:
        yr = trial.get("year", 0)
        year_groups[yr].append(trial)

    for year in sorted(year_groups.keys()):
        for trial in year_groups[year]:
            sponsor = trial.get("sponsor", "Unknown")
            locations = trial.get("locations", [])
            entities = [sponsor]
            for loc in locations:
                facility = loc.get("facility", "")
                country = loc.get("country", "")
                if facility:
                    entities.append(facility)
                elif country:
                    entities.append(country)

            all_nodes.update(entities)

            # Edges: sponsor to each facility
            for i in range(1, len(entities)):
                edge = tuple(sorted([entities[0], entities[i]]))
                all_edges.add(edge)
            # Facility-facility edges within same trial
            for i in range(1, len(entities)):
                for j in range(i + 1, len(entities)):
                    edge = tuple(sorted([entities[i], entities[j]]))
                    all_edges.add(edge)

        n_nodes = len(all_nodes)
        n_edges = len(all_edges)
        max_edges = n_nodes * (n_nodes - 1) / 2 if n_nodes > 1 else 1
        density = n_edges / max_edges if max_edges > 0 else 0.0

        snapshots.append({
            "year": year,
            "n_nodes": n_nodes,
            "n_edges": n_edges,
            "density": round(density, 6),
        })
        years_seen.append(year)

    # Growth rate: average annual node growth
    if len(snapshots) >= 2:
        node_counts = [s["n_nodes"] for s in snapshots]
        rates = []
        for i in range(1, len(node_counts)):
            if node_counts[i - 1] > 0:
                rates.append((node_counts[i] - node_counts[i - 1]) / node_counts[i - 1])
        growth_rate = float(np.mean(rates)) if rates else 0.0
    else:
        growth_rate = 0.0

    # Preferential attachment: fit alpha from degree-new-edge correlation
    # P(new edge to v) ~ deg(v)^alpha
    # Use log-log regression on cumulative degree vs attachment counts
    pa_alpha = 0.0
    if len(snapshots) >= 2:
        # Build degree distribution at each step and measure attachment
        node_degree = defaultdict(int)
        edges_so_far = set()
        degree_vals = []
        attach_counts = []

        for trial in sorted_trials:
            sponsor = trial.get("sponsor", "Unknown")
            locations = trial.get("locations", [])
            entities = [sponsor]
            for loc in locations:
                facility = loc.get("facility", "")
                if facility:
                    entities.append(facility)

            # Record degrees BEFORE adding new edges
            for ent in entities:
                if node_degree[ent] > 0:
                    degree_vals.append(node_degree[ent])
                    attach_counts.append(1)

            # Add edges
            for i in range(len(entities)):
                for j in range(i + 1, len(entities)):
                    edge = tuple(sorted([entities[i], entities[j]]))
                    if edge not in edges_so_far:
                        edges_so_far.add(edge)
                        node_degree[entities[i]] += 1
                        node_degree[entities[j]] += 1

        if len(degree_vals) > 2:
            # Bin by degree and compute average attachment
            deg_arr = np.array(degree_vals, dtype=float)
            att_arr = np.array(attach_counts, dtype=float)
            unique_degs = np.unique(deg_arr)
            if len(unique_degs) > 1:
                mean_attach = []
                for d in unique_degs:
                    mask = deg_arr == d
                    mean_attach.append(float(np.sum(att_arr[mask])))
                log_deg = np.log(unique_degs + 1)
                log_att = np.log(np.array(mean_attach) + 1)
                if len(log_deg) > 1:
                    slope, _, _, _, _ = sp_stats.linregress(log_deg, log_att)
                    pa_alpha = float(slope)

    # Densification exponent: e ~ n^beta
    # log(e) = beta * log(n) + const
    densification_beta = 0.0
    if len(snapshots) >= 2:
        log_n = []
        log_e = []
        for s in snapshots:
            if s["n_nodes"] > 1 and s["n_edges"] > 0:
                log_n.append(math.log(s["n_nodes"]))
                log_e.append(math.log(s["n_edges"]))
        if len(log_n) > 1:
            slope, _, _, _, _ = sp_stats.linregress(log_n, log_e)
            densification_beta = float(slope)

    return {
        "snapshots": snapshots,
        "growth_rate": float(growth_rate),
        "preferential_attachment_alpha": float(pa_alpha),
        "densification_exponent": float(densification_beta),
    }


# ──────────────────────────────────────────────────────────────────────
# 19. Network Entropy
# ──────────────────────────────────────────────────────────────────────

def network_entropy(graph):
    """Compute multiple entropy measures for a graph.

    Args:
        graph: TrialAtlas graph dict.

    Returns:
        dict with:
            structural_entropy: Shannon entropy of degree distribution.
            von_neumann_entropy: H_vn = -Tr(rho log2(rho)) where rho = L/Tr(L).
            graph_energy: E = sum(|eigenvalues of A|).
            complexity_index: C = H * (1 - H/H_max).
    """
    node_list, name_to_idx, adj, degrees, total_weight = _build_adj(graph)
    n = len(node_list)

    if n < 2:
        return {
            "structural_entropy": 0.0,
            "von_neumann_entropy": 0.0,
            "graph_energy": 0.0,
            "complexity_index": 0.0,
        }

    # ── Structural entropy: Shannon on degree distribution ──
    deg_list = [degrees.get(i, 0) for i in range(n)]
    total_deg = sum(deg_list)
    if total_deg > 0:
        p_k = np.array(deg_list) / total_deg
    else:
        p_k = np.ones(n) / n

    structural_entropy = 0.0
    for p in p_k:
        if p > 0:
            structural_entropy -= p * math.log(p)

    # ── Von Neumann entropy ──
    A = np.zeros((n, n))
    for i in adj:
        for j, w in adj[i].items():
            A[i][j] = w

    D_diag = np.array([degrees.get(i, 0.0) for i in range(n)])
    L = np.diag(D_diag) - A

    trace_L = float(np.trace(L))
    if trace_L > 0:
        rho = L / trace_L
        rho = (rho + rho.T) / 2.0  # symmetrize
        eigenvalues_rho = np.linalg.eigvalsh(rho)
        von_neumann_entropy = 0.0
        for ev in eigenvalues_rho:
            if ev > 1e-15:
                von_neumann_entropy -= ev * math.log2(ev)
    else:
        von_neumann_entropy = 0.0

    # ── Graph energy ──
    eigenvalues_A = np.linalg.eigvalsh(A)
    graph_energy = float(np.sum(np.abs(eigenvalues_A)))

    # ── Complexity index: C = H * (1 - H/H_max) ──
    H = structural_entropy
    H_max = math.log(n) if n > 1 else 1.0
    if H_max > 0:
        complexity_index = H * (1.0 - H / H_max)
    else:
        complexity_index = 0.0

    return {
        "structural_entropy": float(structural_entropy),
        "von_neumann_entropy": float(von_neumann_entropy),
        "graph_energy": float(graph_energy),
        "complexity_index": float(complexity_index),
    }


# ──────────────────────────────────────────────────────────────────────
# 20. Hyperbolic Embedding (Poincare disk via stress minimization)
# ──────────────────────────────────────────────────────────────────────

def hyperbolic_embedding(graph, n_dims=2, seed=42):
    """Embed graph nodes into the Poincare disk model of hyperbolic space.

    Uses stress minimization with Riemannian gradient descent in the
    Poincare disk. The hyperbolic distance between points u, v is:
        d_H(u,v) = arccosh(1 + 2 ||u-v||^2 / ((1-||u||^2)(1-||v||^2)))

    Args:
        graph: TrialAtlas graph dict.
        n_dims: Embedding dimensionality (default 2: polar coords r, theta).
        seed: Random seed for reproducibility.

    Returns:
        dict with:
            embeddings: {node: [r, theta, ...]} polar coordinates
            distortion: mean |d_embedded - d_graph| / d_graph
            map_score: mean average precision (fraction of true neighbors
                       recovered in embedding-nearest-neighbor ranking)
    """
    rng = np.random.RandomState(seed)
    node_list, name_to_idx, adj, degrees, total_weight = _build_adj(graph)
    n = len(node_list)

    if n < 2:
        emb = {}
        if n == 1:
            emb[node_list[0]] = [0.0] * n_dims
        return {"embeddings": emb, "distortion": 0.0, "map_score": 1.0}

    # ── BFS shortest-path distances ──
    dist_matrix = np.full((n, n), float('inf'))
    for start in range(n):
        dist_matrix[start][start] = 0.0
        queue = [start]
        visited = {start}
        while queue:
            current = queue.pop(0)
            for nbr in adj.get(current, {}):
                if nbr not in visited:
                    visited.add(nbr)
                    dist_matrix[start][nbr] = dist_matrix[start][current] + 1.0
                    queue.append(nbr)

    # Replace inf with max_dist + 1 for disconnected components
    max_finite = float(np.max(dist_matrix[np.isfinite(dist_matrix)]))
    dist_matrix[~np.isfinite(dist_matrix)] = max_finite + 1.0

    # ── Initialize points in Poincare disk (Cartesian coords) ──
    # Points inside unit disk: ||x|| < 1
    coords = rng.randn(n, n_dims) * 0.1  # small initial positions

    # Clip to inside disk
    for i in range(n):
        norm = np.linalg.norm(coords[i])
        if norm >= 0.95:
            coords[i] = coords[i] / norm * 0.9

    def _hyp_dist(u, v):
        """Poincare disk distance."""
        diff_sq = float(np.sum((u - v) ** 2))
        nu = float(np.sum(u ** 2))
        nv = float(np.sum(v ** 2))
        denom = (1.0 - nu) * (1.0 - nv)
        if denom <= 0:
            return 20.0  # large distance for boundary points
        arg = 1.0 + 2.0 * diff_sq / denom
        if arg < 1.0:
            arg = 1.0
        return math.acosh(arg)

    def _riemannian_grad(u, euclidean_grad):
        """Convert Euclidean gradient to Riemannian (Poincare) gradient."""
        nu = float(np.sum(u ** 2))
        factor = ((1.0 - nu) ** 2) / 4.0
        return factor * euclidean_grad

    # ── Gradient descent ──
    lr = 0.01
    n_iter = 300
    for iteration in range(n_iter):
        # Decay learning rate
        current_lr = lr * (1.0 - iteration / n_iter)

        for i in range(n):
            grad_i = np.zeros(n_dims)
            for j in range(n):
                if i == j:
                    continue
                d_target = dist_matrix[i][j]
                d_embed = _hyp_dist(coords[i], coords[j])

                if d_embed < 1e-10:
                    continue

                # Stress gradient: d(d_embed - d_target)^2 / d(coords_i)
                # Approximate Euclidean gradient
                direction = coords[i] - coords[j]
                dir_norm = np.linalg.norm(direction)
                if dir_norm < 1e-10:
                    continue
                direction = direction / dir_norm
                grad_i += 2.0 * (d_embed - d_target) * direction

            # Riemannian gradient
            riem_grad = _riemannian_grad(coords[i], grad_i)
            coords[i] -= current_lr * riem_grad

            # Project back into disk
            norm = np.linalg.norm(coords[i])
            if norm >= 0.99:
                coords[i] = coords[i] / norm * 0.98

    # ── Convert to polar coordinates ──
    embeddings = {}
    for i, name in enumerate(node_list):
        r = float(np.linalg.norm(coords[i]))
        if n_dims >= 2:
            theta = float(math.atan2(coords[i][1], coords[i][0]))
        else:
            theta = 0.0
        polar = [r, theta]
        # Additional dims as extra angles
        for d in range(2, n_dims):
            polar.append(float(coords[i][d]))
        embeddings[name] = polar[:n_dims]

    # ── Distortion ──
    distortion_sum = 0.0
    distortion_count = 0
    for i in range(n):
        for j in range(i + 1, n):
            d_g = dist_matrix[i][j]
            d_e = _hyp_dist(coords[i], coords[j])
            if d_g > 0:
                distortion_sum += abs(d_e - d_g) / d_g
                distortion_count += 1
    distortion = distortion_sum / distortion_count if distortion_count > 0 else 0.0

    # ── MAP score (mean average precision) ──
    # For each node, rank neighbors by embedding distance, check if true
    # graph neighbors appear first
    map_sum = 0.0
    for i in range(n):
        true_neighbors = set(adj.get(i, {}).keys())
        if not true_neighbors:
            map_sum += 1.0  # no neighbors to miss
            continue

        # Rank all other nodes by hyperbolic distance
        dists = []
        for j in range(n):
            if j == i:
                continue
            dists.append((j, _hyp_dist(coords[i], coords[j])))
        dists.sort(key=lambda x: x[1])

        # Average precision
        hits = 0
        ap_sum = 0.0
        for rank, (j, _) in enumerate(dists, 1):
            if j in true_neighbors:
                hits += 1
                ap_sum += hits / rank
        ap = ap_sum / len(true_neighbors) if true_neighbors else 0.0
        map_sum += ap

    map_score = map_sum / n

    return {
        "embeddings": embeddings,
        "distortion": float(distortion),
        "map_score": float(map_score),
    }


# ──────────────────────────────────────────────────────────────────────
# 21. Network Optimal Transport
# ──────────────────────────────────────────────────────────────────────

def network_optimal_transport(graph1, graph2):
    """Compare two graphs via optimal transport distances.

    Computes:
    1. Wasserstein distance on degree distributions.
    2. Gromov-Wasserstein distance via Sinkhorn on shortest-path
       distance matrices.

    Args:
        graph1, graph2: TrialAtlas graph dicts.

    Returns:
        dict with:
            wasserstein_distance: W1 on degree distributions.
            gromov_wasserstein: approximate GW distance.
            matched_nodes: list of (node1, node2, transport_mass) tuples.
    """
    # ── Helper: extract degree distribution ──
    def _degree_dist(graph):
        """Return sorted degree list."""
        node_list, _, adj, degrees, _ = _build_adj(graph)
        n = len(node_list)
        return node_list, [degrees.get(i, 0) for i in range(n)]

    nodes1, degs1 = _degree_dist(graph1)
    nodes2, degs2 = _degree_dist(graph2)
    n1 = len(nodes1)
    n2 = len(nodes2)

    if n1 == 0 or n2 == 0:
        return {
            "wasserstein_distance": 0.0,
            "gromov_wasserstein": 0.0,
            "matched_nodes": [],
        }

    # ── 1. Wasserstein on degree distributions (W1) ──
    # Sort and compute earth-mover's distance on empirical CDFs
    sorted_d1 = sorted(degs1)
    sorted_d2 = sorted(degs2)

    # Interpolate to same length
    max_len = max(n1, n2)
    interp_d1 = np.interp(
        np.linspace(0, 1, max_len),
        np.linspace(0, 1, n1),
        sorted_d1,
    )
    interp_d2 = np.interp(
        np.linspace(0, 1, max_len),
        np.linspace(0, 1, n2),
        sorted_d2,
    )
    wasserstein = float(np.mean(np.abs(interp_d1 - interp_d2)))

    # ── 2. Gromov-Wasserstein via Sinkhorn ──
    # BFS shortest-path distance matrices
    def _bfs_dist_matrix(graph):
        node_list, _, adj, _, _ = _build_adj(graph)
        n = len(node_list)
        D = np.zeros((n, n))
        for start in range(n):
            visited = {start}
            queue = [start]
            while queue:
                curr = queue.pop(0)
                for nbr in adj.get(curr, {}):
                    if nbr not in visited:
                        visited.add(nbr)
                        D[start][nbr] = D[start][curr] + 1
                        queue.append(nbr)
            # Unreachable: set to n (large)
            for j in range(n):
                if j != start and j not in visited:
                    D[start][j] = n
        return D

    D1 = _bfs_dist_matrix(graph1)
    D2 = _bfs_dist_matrix(graph2)

    # Uniform distributions
    p = np.ones(n1) / n1
    q = np.ones(n2) / n2

    # Cost matrix for GW: C[i,j] = sum_kl |D1[i,k] - D2[j,l]|^2 * p[k] * q[l]
    # This is O(n1*n2*n1*n2), simplify for small graphs
    if n1 * n2 <= 10000:
        C = np.zeros((n1, n2))
        for i in range(n1):
            for j in range(n2):
                val = 0.0
                for k in range(n1):
                    for l in range(n2):
                        val += (D1[i, k] - D2[j, l]) ** 2 * p[k] * q[l]
                C[i, j] = val
    else:
        # Approximate: use random sampling
        rng = np.random.RandomState(42)
        n_samples = 100
        C = np.zeros((n1, n2))
        for _ in range(n_samples):
            k = rng.randint(0, n1)
            l = rng.randint(0, n2)
            for i in range(n1):
                for j in range(n2):
                    C[i, j] += (D1[i, k] - D2[j, l]) ** 2 * p[k] * q[l]
        C *= (n1 * n2) / n_samples

    # Sinkhorn iterations
    reg = 0.1  # entropic regularization
    K = np.exp(-C / reg)
    K = np.maximum(K, 1e-100)

    u = np.ones(n1)
    v = np.ones(n2)

    for _ in range(100):
        u = p / (K @ v)
        v = q / (K.T @ u)
        # Clamp for stability
        u = np.clip(u, 1e-10, 1e10)
        v = np.clip(v, 1e-10, 1e10)

    # Transport plan
    T = np.diag(u) @ K @ np.diag(v)
    gromov_wasserstein = float(np.sum(T * C))

    # ── Matched nodes: top assignments from transport plan ──
    matched_nodes = []
    # For each node in graph1, find best match in graph2
    for i in range(n1):
        j_best = int(np.argmax(T[i, :]))
        matched_nodes.append((nodes1[i], nodes2[j_best], float(T[i, j_best])))

    return {
        "wasserstein_distance": float(wasserstein),
        "gromov_wasserstein": float(gromov_wasserstein),
        "matched_nodes": matched_nodes,
    }
