"""Pluggable matrix-factorization backends for mosaicMPI.

mosaicMPI's :class:`mosaicmpi.cnmf.cNMF` engine owns the whole consensus,
refit, k-selection and on-disk-output machinery. The *only* algorithm-specific
step is the per-iteration factorization of a single (cells x genes) matrix into
a spectra matrix (programs x genes) and a usage matrix (cells x programs).

This module lets that single step be swapped for an alternative implementation
(e.g. the optimal-transport factorization in the ``spot-nmf`` package) while
every other part of the pipeline -- and, critically, the output file format read
back by :meth:`mosaicmpi.dataset.Dataset.add_cnmf_results` -- stays identical.

A backend is just a callable with the signature::

    factorizer(X, n_components, random_state, var_names=None, params=None)
        -> (spectra, usages)

where

* ``X``            : ndarray / sparse matrix, cells x genes (non-negative)
* ``n_components`` : int, the rank k
* ``random_state`` : int seed for reproducibility
* ``var_names``    : optional gene labels (columns of ``X``)
* ``params``       : optional dict of backend-specific keyword arguments
* returns ``spectra`` (k x genes) and ``usages`` (cells x k) as ndarrays,
  matching the orientation of scikit-learn's ``non_negative_factorization``
  return value ``(usages, spectra, n_iter)``.

The built-in ``"cnmf"`` backend is registered lazily as ``None`` -- a sentinel
telling :meth:`cNMF.factorize` to use its own scikit-learn code path unchanged.
External packages register their own backend by calling
:func:`register_factorizer`, either eagerly at import time or via the plugin
autoload map below.
"""

import importlib
import logging

# name -> callable (or None for the built-in scikit-learn/cNMF path)
_BACKENDS = {
    "cnmf": None,
}

# Optional backends that live in separate packages. When an unknown algorithm is
# requested, we try to import the mapped module; that module is expected to call
# register_factorizer() at import time. This keeps the dependency optional and
# one-directional (mosaicMPI does not import these packages unless asked).
_PLUGIN_MODULES = {
    "spotnmf": "spotnmf.mosaic",
}


def register_factorizer(name, factorizer):
    """Register a factorization backend under ``name``.

    :param name: Identifier used by ``--algorithm`` / ``initialize_cnmf(algorithm=...)``.
    :type name: str
    :param factorizer: Callable implementing the backend contract, or ``None``
        to select the built-in scikit-learn/cNMF path.
    :type factorizer: callable or None
    """
    _BACKENDS[name] = factorizer
    logging.info(f"Registered factorization backend: '{name}'")


def available_factorizers():
    """Return the list of currently registered backend names."""
    return sorted(_BACKENDS)


def get_factorizer(name):
    """Return the backend callable registered under ``name``.

    Returns ``None`` for the built-in ``"cnmf"`` path. If ``name`` is not yet
    registered but is a known plugin, the plugin module is imported (which is
    expected to self-register) before looking it up again.

    :raises ValueError: if ``name`` is unknown and no plugin provides it.
    """
    if name not in _BACKENDS and name in _PLUGIN_MODULES:
        module = _PLUGIN_MODULES[name]
        try:
            importlib.import_module(module)
        except ImportError as e:
            raise ValueError(
                f"Factorization backend '{name}' requires the module '{module}', "
                f"which could not be imported: {e}. Install the corresponding "
                f"package (e.g. `pip install spot-nmf`) to use this backend."
            ) from e
    if name not in _BACKENDS:
        raise ValueError(
            f"Unknown factorization backend '{name}'. "
            f"Available: {', '.join(available_factorizers())}."
        )
    return _BACKENDS[name]
