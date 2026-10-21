# Non-interactive backend for the whole test session: compare_models() ends
# with plt.show(), which blocks on an open window under a desktop backend
# (e.g. macOS) and hangs the suite.
import matplotlib

matplotlib.use("Agg")
