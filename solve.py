#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# flake8: noqa
# pylint: disable=broad-exception-raised, raise-missing-from, too-many-arguments, redefined-outer-name
# pylint: disable=multiple-statements, logging-fstring-interpolation, trailing-whitespace, line-too-long
# pylint: disable=broad-exception-caught, missing-function-docstring, missing-class-docstring
# pylint: disable=f-string-without-interpolation
# pylance: disable=reportMissingImports, reportMissingModuleSource
# type: ignore

import json
import argparse
import os
import numpy as np
from scipy.optimize import nnls
import logging
import itertools
from typing import Dict, List, Tuple, Any, Optional, Set

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)



def find_variants(materials: List[Dict], target_umf: Dict[str, float],  max_solutions: int = 10, verbose: bool = False,
                     error_threshold: float = 0.05) -> List[Dict]:
