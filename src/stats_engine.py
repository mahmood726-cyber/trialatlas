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
