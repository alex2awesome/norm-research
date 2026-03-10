# Silence noisy docstring escape warnings in metric definitions
import warnings

warnings.filterwarnings(
    "ignore",
    category=SyntaxWarning,
    message=r".*invalid escape sequence.*",
    module=r"autometrics\.metrics\..*",
)

# Silence pyemd pkg_resources deprecation warning
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=r".*pkg_resources is deprecated as an API.*",
)
