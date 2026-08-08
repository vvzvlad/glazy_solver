#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# flake8: noqa
# pylint: disable=broad-exception-raised, raise-missing-from, too-many-arguments, redefined-outer-name
# pylint: disable=multiple-statements, logging-fstring-interpolation, trailing-whitespace, line-too-long
# pylint: disable=broad-exception-caught, missing-function-docstring, missing-class-docstring
# pylint: disable=f-string-without-interpolation
# pylance: disable=reportMissingImports, reportMissingModuleSource
# type: ignore

# Import of a public recipe from glazy.org.
#
# Only the target formula and the original recipe are taken: no attempt is made
# to map the materials of Glazy onto database/materials.json, the solver picks
# its own materials for the imported UMF.
#
# The target UMF is recomputed from the WEIGHT analysis (percentageAnalysis) by
# common.weights_to_umf(), it is not the umfAnalysis of Glazy. The two are
# expressed on different unity bases: common.oxides_classification() counts
# FeO/CoO/NiO/CuO as fluxes and knows no PbO, modern Glazy does the opposite
# (see TZ_SOLVER_V2.md, section 7.2 on the flux conventions). On a recipe
# carrying such colorants the numbers diverge by up to ~20% on every oxide at
# once, and a target in a basis the solver does not use could never be matched.
# The conversion itself is scale invariant, so it does not matter that
# percentageAnalysis does not add up to 100 (LOI is excluded from it).
# On a recipe without those colorants both paths agree to ~0.0006 (measured on
# recipe 72382), so nothing is lost in the common case. The numbers of Glazy are
# still returned as umf_glazy, for display next to ours.
#
# This module stays Flask free: it raises GlazyImportError and the API layer
# turns that into the JSON error format of the project.

import html
import math
import re
import requests
from common import load_molar_masses, weights_to_umf

GLAZY_API_BASE = 'https://api.glazy.org/api'
GLAZY_RECIPE_URL = 'https://glazy.org/recipes/{recipe_id}'

# Glazy serves the anonymous API to its own frontend; Origin/Referer of that
# frontend are what makes the request pass, Accept keeps the answer JSON
GLAZY_HEADERS = {
    'Accept': 'application/json',
    'Origin': 'https://glazy.org',
    'Referer': 'https://glazy.org/',
}

# (connect, read). A host that does not accept a connection within 5 s is down
# and waiting longer only delays the error; the read budget is much larger
# because Glazy answers a cold recipe slowly.
HTTP_TIMEOUT = (5, 20)

# Where the returned target UMF comes from. The UI needs to know: "glazy_umf"
# means the target is in the flux basis of Glazy and not in ours.
UMF_SOURCE_WEIGHTS = 'weights'
UMF_SOURCE_GLAZY = 'glazy_umf'

# A URL keeps the id after "recipes/"; the slug, the query and the fragment that
# may follow are irrelevant. This form is matched BEFORE the bare-digits one so
# that a URL carrying other numbers (a material id, a year in a slug) can never
# be read as an id itself.
_RECIPE_URL_RE = re.compile(r'recipes/(\d+)')
_BARE_ID_RE = re.compile(r'\A(\d+)\Z')


class GlazyImportError(Exception):
    """
    Failure of the import that the caller is expected to report to the user

    Args:
        code: short snake_case identifier for the "error" field of the JSON answer
        message: human readable explanation
        http_status: HTTP status the API layer should answer with
    """

    def __init__(self, code, message, http_status):
        super().__init__(message)
        self.code = code
        self.message = message
        self.http_status = http_status


def parse_recipe_id(value):
    """
    Extract the Glazy recipe id from whatever the user pasted

    Args:
        value: recipe id as a number or a string, or a glazy.org recipe URL

    Returns:
        positive int id, or None when the value is not a recipe reference
    """
    # bool is a subclass of int, and True would silently become recipe 1
    if isinstance(value, bool):
        return None

    if isinstance(value, int):
        return value if value > 0 else None

    if not isinstance(value, str):
        return None

    text = value.strip()
    if not text:
        return None

    match = _RECIPE_URL_RE.search(text)
    if match is None:
        match = _BARE_ID_RE.match(text)
    if match is None:
        return None

    recipe_id = int(match.group(1))
    return recipe_id if recipe_id > 0 else None


def _error_for_status(status_code, message):
    """
    Build the GlazyImportError matching a Glazy status code

    Args:
        status_code: status reported by Glazy (in the body or in the response)
        message: message reported by Glazy, may be empty

    Returns:
        GlazyImportError ready to be raised
    """
    if status_code == 404:
        return GlazyImportError('glazy_not_found', message or 'Recipe does not exist', 404)

    # Anonymous access is the whole point of this import, so 401 and 403 mean
    # the same thing for the user: the recipe is not public
    if status_code in (401, 403):
        return GlazyImportError(
            'glazy_forbidden',
            'Recipe is not public: the import works with public Glazy recipes only',
            403
        )

    return GlazyImportError('glazy_unavailable', message or f'Glazy API error, status {status_code}', 502)


def _read_error_body(error_body, http_status):
    """
    Read the status and the message out of an error body of Glazy

    Args:
        error_body: value of the "error" key, a dict or a bare string
        http_status: HTTP status of the response, used when the body has none

    Returns:
        (status_code, message)
    """
    if isinstance(error_body, dict):
        try:
            status_code = int(error_body.get('status_code', http_status))
        except (TypeError, ValueError):
            status_code = http_status
        return status_code, str(error_body.get('message') or '')

    return http_status, str(error_body)


def fetch_recipe(recipe_id):
    """
    Fetch a public recipe from the Glazy API

    Args:
        recipe_id: numeric Glazy recipe id

    Returns:
        the "data" dictionary of the answer

    Raises:
        GlazyImportError: on any transport, protocol or access failure
    """
    url = f'{GLAZY_API_BASE}/recipes/{recipe_id}'

    try:
        response = requests.get(url, headers=GLAZY_HEADERS, timeout=HTTP_TIMEOUT)
    except requests.Timeout:
        raise GlazyImportError('glazy_timeout', 'Glazy did not answer in time', 504)
    except requests.RequestException as exc:
        raise GlazyImportError('glazy_unavailable', f'Glazy is not reachable: {exc}', 502)

    try:
        payload = response.json()
    except ValueError:
        raise GlazyImportError('glazy_unavailable', 'Glazy returned a non-JSON answer', 502)

    # Glazy answers HTTP 200 even for "recipe does not exist" and for a private
    # recipe, with the real status inside the body - so the body is inspected
    # first and the HTTP status is only the fallback
    if isinstance(payload, dict) and 'error' in payload:
        status_code, message = _read_error_body(payload['error'], response.status_code)
        raise _error_for_status(status_code, message)

    if response.status_code >= 400:
        raise _error_for_status(response.status_code, '')

    data = payload.get('data') if isinstance(payload, dict) else None
    if not isinstance(data, dict) or not data:
        raise GlazyImportError('glazy_unavailable', 'Glazy returned an answer without recipe data', 502)

    return data


def _numeric_oxides(analysis):
    """
    Turn a raw analysis of Glazy into {oxide: float}

    Every value of Glazy is a string, and the analyses carry derived entries
    next to the oxides: SiO2Al2O3Ratio, R2OTotal, ROTotal, xAl2O3 in umfAnalysis
    and loi in percentageAnalysis. Keeping only the keys of the molar mass table
    drops all of them without a blacklist that would need maintenance whenever
    Glazy adds another one.

    Args:
        analysis: raw analysis dictionary of Glazy, may be None

    Returns:
        dictionary {oxide: float} with the finite positive values only
    """
    if not isinstance(analysis, dict):
        return {}

    molar_masses = load_molar_masses()

    oxides = {}
    for oxide, raw_value in analysis.items():
        if oxide not in molar_masses:
            continue
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value) and value > 0:
            oxides[oxide] = value

    return oxides


def _optional_float(raw_value):
    """Parse a numeric string of Glazy, returning None when it is absent or unusable"""
    try:
        value = float(raw_value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _optional_int(raw_value):
    """Parse an id of Glazy, returning None when it is absent or unusable"""
    if isinstance(raw_value, bool):
        return None
    try:
        return int(raw_value)
    except (TypeError, ValueError):
        return None


def _cone_name(raw_value):
    """
    Normalize an Orton cone name of Glazy

    Glazy stores the names HTML escaped ("5&#189;" for 5 1/2). They are
    unescaped here, on the server, so that every consumer can treat them as
    plain text: the UI inserts them with textContent and must never have to
    interpret markup coming from a third party.

    Args:
        raw_value: cone name as returned by Glazy, may be None

    Returns:
        unescaped cone name, or None when absent
    """
    if raw_value is None:
        return None

    name = html.unescape(str(raw_value)).strip()
    return name or None


def extract_components(recipe_data):
    """
    Extract the material list of the original Glazy recipe

    Args:
        recipe_data: the "data" dictionary of a Glazy answer

    Returns:
        list of {"name", "percentage", "is_additional", "glazy_material_id"} in
        the order of the recipe; entries with an unusable percentage are skipped
    """
    raw_components = recipe_data.get('materialComponents')
    if not isinstance(raw_components, list):
        return []

    components = []
    for entry in raw_components:
        if not isinstance(entry, dict):
            continue

        percentage = _optional_float(entry.get('percentageAmount'))
        if percentage is None:
            continue

        material = entry.get('material')
        if not isinstance(material, dict):
            material = {}

        components.append({
            'name': str(material.get('name') or ''),
            'percentage': percentage,
            'is_additional': bool(entry.get('isAdditional')),
            'glazy_material_id': _optional_int(material.get('id')),
        })

    return components


def build_import_result(recipe_data, recipe_id=None):
    """
    Build the answer of the import endpoint from a Glazy recipe

    Args:
        recipe_data: the "data" dictionary of a Glazy answer
        recipe_id: id that was requested, used when the payload carries none

    Returns:
        dictionary with the target umf, the analyses of Glazy, the original
        material list and the firing information

    Raises:
        GlazyImportError: when the recipe carries no usable oxide analysis
    """
    analysis = recipe_data.get('analysis')
    if not isinstance(analysis, dict):
        analysis = {}

    weight_percent = _numeric_oxides(analysis.get('percentageAnalysis'))
    umf_glazy = _numeric_oxides(analysis.get('umfAnalysis'))

    if weight_percent:
        umf = weights_to_umf(weight_percent)
        umf_source = UMF_SOURCE_WEIGHTS
    elif umf_glazy:
        # Nothing to recompute from: the target is taken as is, in the basis of
        # Glazy. umf_source says so, the UI warns about it.
        umf = dict(umf_glazy)
        umf_source = UMF_SOURCE_GLAZY
    else:
        raise GlazyImportError('no_analysis', 'Glazy recipe has no oxide analysis to import', 422)

    result_id = _optional_int(recipe_data.get('id'))
    if result_id is None:
        result_id = recipe_id

    return {
        'id': result_id,
        'name': str(recipe_data.get('name') or ''),
        'url': GLAZY_RECIPE_URL.format(recipe_id=result_id) if result_id is not None else None,
        'umf': umf,
        'umf_source': umf_source,
        'umf_glazy': umf_glazy,
        'weight_percent': weight_percent,
        'components': extract_components(recipe_data),
        'cone_from': _cone_name(recipe_data.get('fromOrtonConeName')),
        'cone_to': _cone_name(recipe_data.get('toOrtonConeName')),
        'thermal_expansion': _optional_float(analysis.get('thermalExpansion')),
    }
