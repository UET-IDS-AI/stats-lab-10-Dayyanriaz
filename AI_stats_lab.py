import numpy as np

# -------------------------------------------------
# Question 1: Joint Gaussian PDF and Marginals
# -------------------------------------------------

def joint_gaussian_pdf(x, y, mu_x=1, mu_y=-2, sigma_x=2, sigma_y=3, rho=0.6):
    norm = 1.0 / (2 * np.pi * sigma_x * sigma_y * np.sqrt(1 - rho**2))
    Q = (
        ((x - mu_x)**2) / sigma_x**2
        - 2 * rho * ((x - mu_x) * (y - mu_y)) / (sigma_x * sigma_y)
        + ((y - mu_y)**2) / sigma_y**2
    )
    return norm * np.exp(-Q / (2 * (1 - rho**2)))


def marginal_pdf_x(x, mu_x=1, sigma_x=2):
    return (1.0 / (np.sqrt(2 * np.pi) * sigma_x)) * np.exp(-0.5 * ((x - mu_x) / sigma_x)**2)


def marginal_pdf_y(y, mu_y=-2, sigma_y=3):
    return (1.0 / (np.sqrt(2 * np.pi) * sigma_y)) * np.exp(-0.5 * ((y - mu_y) / sigma_y)**2)


def covariance_matrix(sigma_x=2, sigma_y=3, rho=0.6):
    return np.array([
        [sigma_x**2,          rho * sigma_x * sigma_y],
        [rho * sigma_x * sigma_y, sigma_y**2         ]
    ])


def joint_pdf_grid_integral(mu_x=1, mu_y=-2, sigma_x=2, sigma_y=3, rho=0.6, n=250):
    x = np.linspace(mu_x - 4*sigma_x, mu_x + 4*sigma_x, n)
    y = np.linspace(mu_y - 4*sigma_y, mu_y + 4*sigma_y, n)
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    xx, yy = np.meshgrid(x, y)
    z = joint_gaussian_pdf(xx, yy, mu_x, mu_y, sigma_x, sigma_y, rho)
    return np.sum(z) * dx * dy


# -------------------------------------------------
# Question 2: Simulation and Independence
# -------------------------------------------------

def generate_joint_gaussian_samples(
    n=100000, mu_x=1, mu_y=-2, sigma_x=2, sigma_y=3, rho=0.6, seed=0
):
    rng = np.random.default_rng(seed)
    mean = [mu_x, mu_y]
    cov = covariance_matrix(sigma_x, sigma_y, rho)
    samples = rng.multivariate_normal(mean, cov, size=n)
    return samples[:, 0], samples[:, 1]


def sample_means(x_samples, y_samples):
    return np.mean(x_samples), np.mean(y_samples)


def sample_covariance_matrix(x_samples, y_samples):
    data = np.vstack([x_samples, y_samples])   # shape (2, n)
    return np.cov(data, ddof=1)


def sample_correlation(x_samples, y_samples):
    cov = sample_covariance_matrix(x_samples, y_samples)
    return cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])


def gaussian_independence_check(rho):
    return rho == 0


def zero_rho_covariance_check(n=100000):
    x, y = generate_joint_gaussian_samples(n=n, rho=0, seed=1)
    cov = sample_covariance_matrix(x, y)
    return bool(abs(cov[0, 1]) < 0.1)


def nonzero_rho_covariance_check(n=100000):
    rho, sigma_x, sigma_y = 0.6, 2, 3
    x, y = generate_joint_gaussian_samples(n=n, rho=rho, seed=1)
    cov = sample_covariance_matrix(x, y)
    expected = rho * sigma_x * sigma_y   # 3.6
    return bool(abs(cov[0, 1] - expected) < 0.2)
